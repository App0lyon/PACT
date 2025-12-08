from __future__ import annotations

import torch
import torch.nn as nn

from pact import PACTActivation
from quant import QuantConv2d, QuantLinear


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, act_bits: int | None = 4, w_bits: int | None = 4):
        super().__init__()
        self.conv1 = QuantConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, w_bits=w_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuantConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, w_bits=w_bits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act1 = PACTActivation(num_bits=act_bits)
        self.act2 = PACTActivation(num_bits=act_bits)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                QuantConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False, w_bits=w_bits),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act2(out)
        return out


def _make_layer(block, in_planes: int, planes: int, num_blocks: int, stride: int, act_bits: int | None = 4, w_bits: int | None = 4):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for s in strides:
        layers.append(block(in_planes, planes, s, act_bits=act_bits, w_bits=w_bits))
        in_planes = planes * block.expansion
    return nn.Sequential(*layers), in_planes


class ResNetCIFAR(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes: int = 10,
        act_bits: int | None = 4,
        w_bits: int | None = 4,
        quantize_first_last: bool = False,
    ):
        super().__init__()
        self.in_planes = 16
        first_w_bits = w_bits if quantize_first_last else None
        self.conv1 = QuantConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, w_bits=first_w_bits)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = PACTActivation(num_bits=act_bits if quantize_first_last else None)

        self.layer1, self.in_planes = _make_layer(block, self.in_planes, 16, num_blocks[0], stride=1, act_bits=act_bits, w_bits=w_bits)
        self.layer2, self.in_planes = _make_layer(block, self.in_planes, 32, num_blocks[1], stride=2, act_bits=act_bits, w_bits=w_bits)
        self.layer3, self.in_planes = _make_layer(block, self.in_planes, 64, num_blocks[2], stride=2, act_bits=act_bits, w_bits=w_bits)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        last_w_bits = w_bits if quantize_first_last else None
        self.fc = QuantLinear(64 * block.expansion, num_classes, w_bits=last_w_bits)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet20(
    num_classes: int = 10,
    act_bits: int | None = 4,
    w_bits: int | None = 4,
    quantize_first_last: bool = False,
) -> ResNetCIFAR:
    return ResNetCIFAR(
        BasicBlock,
        [3, 3, 3],
        num_classes=num_classes,
        act_bits=act_bits,
        w_bits=w_bits,
        quantize_first_last=quantize_first_last,
    )
