from __future__ import annotations

import torch
import torch.nn as nn

from pact import PACTActivation
from quant import QuantConv2d, QuantLinear


class AlexNetBN(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        act_bits: int | None = 4,
        w_bits: int | None = 4,
        quantize_first_last: bool = False,
    ):
        super().__init__()
        first_act_bits = act_bits if quantize_first_last else None
        first_w_bits = w_bits if quantize_first_last else None
        last_w_bits = w_bits if quantize_first_last else None
        self.features = nn.Sequential(
            QuantConv2d(3, 64, kernel_size=11, stride=4, padding=2, w_bits=first_w_bits),
            nn.BatchNorm2d(64),
            PACTActivation(num_bits=first_act_bits),
            nn.MaxPool2d(kernel_size=3, stride=2),
            QuantConv2d(64, 192, kernel_size=5, padding=2, w_bits=w_bits),
            nn.BatchNorm2d(192),
            PACTActivation(num_bits=act_bits),
            nn.MaxPool2d(kernel_size=3, stride=2),
            QuantConv2d(192, 384, kernel_size=3, padding=1, w_bits=w_bits),
            nn.BatchNorm2d(384),
            PACTActivation(num_bits=act_bits),
            QuantConv2d(384, 256, kernel_size=3, padding=1, w_bits=w_bits),
            nn.BatchNorm2d(256),
            PACTActivation(num_bits=act_bits),
            QuantConv2d(256, 256, kernel_size=3, padding=1, w_bits=w_bits),
            nn.BatchNorm2d(256),
            PACTActivation(num_bits=act_bits),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            QuantLinear(256 * 6 * 6, 4096, w_bits=w_bits),
            PACTActivation(num_bits=act_bits),
            nn.Dropout(),
            QuantLinear(4096, 4096, w_bits=w_bits),
            PACTActivation(num_bits=act_bits),
            QuantLinear(4096, num_classes, w_bits=last_w_bits),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet_bn(
    num_classes: int = 1000,
    act_bits: int | None = 4,
    w_bits: int | None = 4,
    quantize_first_last: bool = False,
) -> AlexNetBN:
    return AlexNetBN(num_classes=num_classes, act_bits=act_bits, w_bits=w_bits, quantize_first_last=quantize_first_last)
