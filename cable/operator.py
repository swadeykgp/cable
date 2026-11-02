"""CABLE: residual local interaction for 2-D feature maps.

Implementation corresponds to the local operator used in the CABLE study.
"""
import torch
from torch import nn


class CABLE2D(nn.Module):
    """Depthwise 3x3 -> grouped 1x1 -> dense 1x1, plus input residual.

    Parameters
    ----------
    channels : int
        Input and output channel count.
    groups : int
        Number of groups in the intermediate pointwise projection.
    kernel_size : int
        Odd spatial kernel size for depthwise mixing (default: 3).

    The two pointwise projections are linear, with no intervening activation.
    This faithfully preserves the experimental operator; it does not include
    the separately explored pooled-global branch.
    """
    def __init__(self, channels: int, groups: int = 4, kernel_size: int = 3):
        super().__init__()
        if channels <= 0 or groups <= 0 or channels % groups:
            raise ValueError("channels must be positive and divisible by groups")
        if kernel_size <= 0 or kernel_size % 2 != 1:
            raise ValueError("kernel_size must be a positive odd integer")
        self.local_mix = nn.Conv2d(channels, channels, kernel_size,
                                   padding=kernel_size // 2,
                                   groups=channels, bias=False)
        self.head_proj = nn.Conv2d(channels, channels, 1,
                                   groups=groups, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.out_proj(self.head_proj(self.local_mix(x)))
        return x + delta
