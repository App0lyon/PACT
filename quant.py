from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def quantize_weight_dorefa(w: torch.Tensor, num_bits: int | None, eps: float = 1e-8) -> torch.Tensor:
    """
    DoReFa-style weight quantization.
    Steps: tanh -> normalize to [0,1] -> k-bit uniform quantization (STE) -> map back to [-1,1].
    """
    if num_bits is None or num_bits >= 32:
        return w
    # Bounded transform.
    w_tanh = torch.tanh(w)
    scale = w_tanh.detach().abs().max().clamp(min=eps)
    w_norm = w_tanh / (2 * scale) + 0.5  # in [0,1]
    levels = (1 << num_bits) - 1
    w_scaled = w_norm * levels
    w_ste = (w_scaled.round() - w_scaled).detach() + w_scaled
    w_q = w_ste / levels
    w_q = 2 * w_q - 1  # back to [-1,1]
    return w_q


class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, w_bits: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_w_bits = w_bits
        self.w_bits = w_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = quantize_weight_dorefa(self.weight, self.w_bits)
        return F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class QuantLinear(nn.Linear):
    def __init__(self, *args, w_bits: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_w_bits = w_bits
        self.w_bits = w_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = quantize_weight_dorefa(self.weight, self.w_bits)
        return F.linear(x, weight, self.bias)
