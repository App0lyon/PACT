from __future__ import annotations

import torch
from torch import nn


class PACTActivation(nn.Module):
    """
    Parameterized Clipping Activation with uniform k-bit quantization.

    y = clip(x, 0, alpha)
    y_q = round(y * (2^k - 1) / alpha) * alpha / (2^k - 1)
    Straight-through estimator (STE) is used for the rounding operation.
    """

    def __init__(self, num_bits: int | None = 4, init_alpha: float = 10.0, eps: float = 1e-6):
        super().__init__()
        self.base_num_bits = num_bits
        self.num_bits = num_bits
        self.eps = eps
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure alpha is positive to avoid division by zero.
        alpha = torch.clamp(self.alpha, min=self.eps).to(dtype=x.dtype, device=x.device)
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        y = torch.clamp(x, min=zero, max=alpha)
        if self.num_bits is None:
            return y

        levels = (1 << self.num_bits) - 1
        # Scale to integer grid.
        y_scaled = y * levels / alpha
        # STE: pretend rounding is identity in backward.
        y_ste = (y_scaled.round() - y_scaled).detach() + y_scaled
        y_q = y_ste * alpha / levels
        return y_q
