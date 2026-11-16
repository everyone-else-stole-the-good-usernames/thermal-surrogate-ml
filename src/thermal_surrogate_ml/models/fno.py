"""2D Fourier Neural Operator (FNO) surrogate.

Operates on channels-first tensors ``[B, C, H, W]``: FFTs act over the spatial
axes, while the linear lift/projection layers act over channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _complex_parameter(size: tuple[int, ...], scale: float) -> torch.Tensor:
    """Sample a complex parameter.

    ``torch.rand`` has no complex dtype, so the real and imaginary parts are
    drawn separately.
    """
    real = torch.rand(size, dtype=torch.float32)
    imag = torch.rand(size, dtype=torch.float32)
    return scale * torch.complex(real, imag)


class SpectralConv2d(nn.Module):
    """Spectral convolution truncated to the lowest Fourier modes.

    Only the low-frequency block is learned; higher modes are zeroed, which is
    what makes the operator resolution-independent.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        self.weights1 = nn.Parameter(_complex_parameter(shape, self.scale))
        self.weights2 = nn.Parameter(_complex_parameter(shape, self.scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transform ``x`` of shape ``[B, C, H, W]`` into ``[B, out, H, W]``."""
        batch = x.shape[0]
        height = x.size(-2)
        width = x.size(-1)

        x_ft = torch.fft.rfft2(x)  # [B, in, H, W // 2 + 1]
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # Low positive modes along both axes.
        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        # Negative modes along the first axis (second axis is rfft-symmetric).
        out_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )

        return torch.fft.irfft2(out_ft, s=(height, width))


class FNO2d(nn.Module):
    """Fourier Neural Operator for 2D field-to-field mapping.

    A channel lift, ``n_layers`` spectral blocks with a pointwise bypass, and a
    two-layer projection back to ``out_channels``.
    """

    def __init__(
        self,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.fc0 = nn.Linear(in_channels, width)

        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(n_layers)]
        )
        self.bypasses = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=1) for _ in range(n_layers)]
        )

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, in, H, W]`` to ``[B, out, H, W]``.

        The linear layers act on channels, so the tensor is transposed to
        ``[B, H, W, C]`` around them.
        """
        # Lift to `width` channels, transposing for the channel-wise linear.
        x = self.fc0(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        for conv, bypass in zip(self.convs, self.bypasses):
            x = F.gelu(conv(x) + bypass(x))

        # Project back to `out_channels`.
        x = self.fc2(F.gelu(self.fc1(x.permute(0, 2, 3, 1))))
        return x.permute(0, 3, 1, 2)