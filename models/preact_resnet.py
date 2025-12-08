from __future__ import annotations

import torch
import torch.nn as nn

from pact import PACTActivation
from quant import QuantConv2d, QuantLinear


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, act_bits: int | None = 4, w_bits: int | None = 4):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = QuantConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, w_bits=w_bits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = QuantConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, w_bits=w_bits)
        self.act1 = PACTActivation(num_bits=act_bits)
        self.act2 = PACTActivation(num_bits=act_bits)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = QuantConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False, w_bits=w_bits)

    def forward(self, x):
        out = self.act1(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(self.act2(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes: int, planes: int, stride: int = 1, act_bits: int | None = 4, w_bits: int | None = 4):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = QuantConv2d(in_planes, planes, kernel_size=1, bias=False, w_bits=w_bits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = QuantConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, w_bits=w_bits)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = QuantConv2d(planes, planes * self.expansion, kernel_size=1, bias=False, w_bits=w_bits)
        self.act1 = PACTActivation(num_bits=act_bits)
        self.act2 = PACTActivation(num_bits=act_bits)
        self.act3 = PACTActivation(num_bits=act_bits)

        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = QuantConv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False, w_bits=w_bits)

    def forward(self, x):
        out = self.act1(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(self.act2(self.bn2(out)))
        out = self.conv3(self.act3(self.bn3(out)))
        out += shortcut
        return out


def _make_layer(block, in_planes: int, planes: int, num_blocks: int, stride: int, act_bits: int | None = 4, w_bits: int | None = 4):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for s in strides:
        layers.append(block(in_planes, planes, s, act_bits=act_bits, w_bits=w_bits))
        in_planes = planes * block.expansion
    return nn.Sequential(*layers), in_planes


class PreActResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes: int = 1000,
        act_bits: int | None = 4,
        w_bits: int | None = 4,
        quantize_first_last: bool = False,
    ):
        super().__init__()
        self.in_planes = 64

        first_w_bits = w_bits if quantize_first_last else None
        last_w_bits = w_bits if quantize_first_last else None
        self.conv1 = QuantConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, w_bits=first_w_bits)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = PACTActivation(num_bits=act_bits if quantize_first_last else None)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1, self.in_planes = _make_layer(block, self.in_planes, 64, num_blocks[0], stride=1, act_bits=act_bits, w_bits=w_bits)
        self.layer2, self.in_planes = _make_layer(block, self.in_planes, 128, num_blocks[1], stride=2, act_bits=act_bits, w_bits=w_bits)
        self.layer3, self.in_planes = _make_layer(block, self.in_planes, 256, num_blocks[2], stride=2, act_bits=act_bits, w_bits=w_bits)
        self.layer4, self.in_planes = _make_layer(block, self.in_planes, 512, num_blocks[3], stride=2, act_bits=act_bits, w_bits=w_bits)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = QuantLinear(512 * block.expansion, num_classes, w_bits=last_w_bits)

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
        x = self.conv1(x)
        x = self.maxpool(self.act1(self.bn1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def preact_resnet18(
    num_classes: int = 1000,
    act_bits: int | None = 4,
    w_bits: int | None = 4,
    quantize_first_last: bool = False,
) -> PreActResNet:
    return PreActResNet(
        PreActBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        act_bits=act_bits,
        w_bits=w_bits,
        quantize_first_last=quantize_first_last,
    )


def preact_resnet50(
    num_classes: int = 1000,
    act_bits: int | None = 4,
    w_bits: int | None = 4,
    quantize_first_last: bool = False,
) -> PreActResNet:
    return PreActResNet(
        PreActBottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        act_bits=act_bits,
        w_bits=w_bits,
        quantize_first_last=quantize_first_last,
    )
