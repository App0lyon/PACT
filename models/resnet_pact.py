import torch
import torch.nn as nn
from models.quant_layers import QuantConv2d, QuantLinear

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bit_width_w=4, bit_width_a=4):
        super().__init__()
        self.conv1 = QuantConv2d(in_planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.conv2 = QuantConv2d(planes, planes, kernel_size=3, stride=1,
                                 padding=1, bit_width_w=bit_width_w, bit_width_a=bit_width_a)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                QuantConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                            bit_width_w=bit_width_w, bit_width_a=bit_width_a),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class ResNetPACT(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bit_width_w=4, bit_width_a=4):
        super().__init__()
        self.in_planes = 16

        self.conv1 = QuantConv2d(3, 16, kernel_size=3, stride=1, padding=1,
                                 bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1,
                                       bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2,
                                       bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2,
                                       bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = QuantLinear(64 * block.expansion, num_classes,
                                  bit_width_w=bit_width_w, bit_width_a=bit_width_a)

    def _make_layer(self, block, planes, num_blocks, stride, bit_width_w, bit_width_a):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, bit_width_w, bit_width_a))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def resnet20_pact(num_classes=10, bit_width_w=4, bit_width_a=4):
    return ResNetPACT(BasicBlock, [3, 3, 3], num_classes, bit_width_w, bit_width_a)

def resnet18_pact(num_classes=1000, bit_width_w=4, bit_width_a=4):
    return ResNetPACT(BasicBlock, [2, 2, 2, 2], num_classes, bit_width_w, bit_width_a)
