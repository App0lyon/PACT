from __future__ import annotations

import torch
import torch.nn as nn

from pact import PACTActivation
from quant import QuantConv2d, QuantLinear


class SVHNConvNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        act_bits: int | None = 4,
        w_bits: int | None = 4,
        quantize_first_last: bool = False,
    ):
        super().__init__()
        first_w_bits = w_bits if quantize_first_last else None
        last_w_bits = w_bits if quantize_first_last else None
        self.conv1 = QuantConv2d(3, 64, kernel_size=3, padding=1, bias=False, w_bits=first_w_bits)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = QuantConv2d(64, 64, kernel_size=3, padding=1, bias=False, w_bits=w_bits)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = QuantConv2d(64, 128, kernel_size=3, padding=1, bias=False, w_bits=w_bits)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = QuantConv2d(128, 128, kernel_size=3, padding=1, bias=False, w_bits=w_bits)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = QuantConv2d(128, 256, kernel_size=3, padding=1, bias=False, w_bits=w_bits)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = QuantConv2d(256, 256, kernel_size=3, padding=1, bias=False, w_bits=w_bits)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = QuantConv2d(256, 512, kernel_size=3, padding=1, bias=False, w_bits=w_bits)
        self.bn7 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = QuantLinear(512 * 4 * 4, num_classes, w_bits=last_w_bits)

        # Activations (share alpha per layer).
        self.act1 = PACTActivation(num_bits=act_bits if quantize_first_last else None)
        self.act2 = PACTActivation(num_bits=act_bits)
        self.act3 = PACTActivation(num_bits=act_bits)
        self.act4 = PACTActivation(num_bits=act_bits)
        self.act5 = PACTActivation(num_bits=act_bits)
        self.act6 = PACTActivation(num_bits=act_bits)
        self.act7 = PACTActivation(num_bits=act_bits if quantize_first_last else None)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.pool(x)

        x = self.act5(self.bn5(self.conv5(x)))
        x = self.act6(self.bn6(self.conv6(x)))
        x = self.pool(x)

        x = self.act7(self.bn7(self.conv7(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def svhn_convnet(
    num_classes: int = 10,
    act_bits: int | None = 4,
    w_bits: int | None = 4,
    quantize_first_last: bool = False,
) -> SVHNConvNet:
    return SVHNConvNet(
        num_classes=num_classes,
        act_bits=act_bits,
        w_bits=w_bits,
        quantize_first_last=quantize_first_last,
    )
