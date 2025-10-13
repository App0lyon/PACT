import torch
import torch.nn as nn
from models.quant_layers import QuantConv2d, QuantLinear
from models.pact_activation import PACTActivation

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bit_width_w=4, bit_width_a=4):
        super().__init__()
        self.conv1 = QuantConv2d(in_planes, planes, kernel_size=3, stride=stride,
                                 padding=1, bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.conv2 = QuantConv2d(planes, planes, kernel_size=3, stride=1,
                                 padding=1, bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.conv2.act = nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                QuantConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                            bit_width_w=bit_width_w, bit_width_a=bit_width_a),
            )
            # remove activation on downsample path to keep it linear
            self.shortcut[0].act = nn.Identity()
        self.final_act = PACTActivation(bit_width=bit_width_a)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.final_act(out)
        return out

class ResNetPACT(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, bit_width_w=4, bit_width_a=4):
        super().__init__()

        if len(num_blocks) == 3:
            self.in_planes = 16
            planes = [16, 32, 64]
            strides = [1, 2, 2]
            self.conv1 = QuantConv2d(3, planes[0], kernel_size=3, stride=1, padding=1,
                                     bit_width_w=None, bit_width_a=None)
            self.maxpool = nn.Identity()
        elif len(num_blocks) == 4:
            self.in_planes = 64
            planes = [64, 128, 256, 512]
            strides = [1, 2, 2, 2]
            self.conv1 = QuantConv2d(3, planes[0], kernel_size=7, stride=2, padding=3,
                                     bit_width_w=None, bit_width_a=None)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError("num_blocks must have length 3 (ResNet20) or 4 (ResNet18).")

        self.layers = nn.ModuleList()
        for idx, (planes_i, blocks_i, stride_i) in enumerate(zip(planes, num_blocks, strides), start=1):
            layer = self._make_layer(block, planes_i, blocks_i, stride_i,
                                     bit_width_w, bit_width_a)
            self.layers.append(layer)
            setattr(self, f"layer{idx}", layer)
        # ensure layer4 attribute exists for CIFAR variant
        for idx in range(len(self.layers) + 1, 5):
            setattr(self, f"layer{idx}", nn.Identity())

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = QuantLinear(planes[-1] * block.expansion, num_classes,
                                  bit_width_w=None, bit_width_a=None)

    def _make_layer(self, block, planes, num_blocks, stride, bit_width_w, bit_width_a):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, bit_width_w, bit_width_a))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        for layer in self.layers:
            out = layer(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def resnet20_pact(num_classes=10, bit_width_w=4, bit_width_a=4):
    return ResNetPACT(BasicBlock, [3, 3, 3], num_classes, bit_width_w, bit_width_a)

def resnet18_pact(num_classes=1000, bit_width_w=4, bit_width_a=4):
    return ResNetPACT(BasicBlock, [2, 2, 2, 2], num_classes, bit_width_w, bit_width_a)
