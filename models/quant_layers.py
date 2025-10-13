import torch
import torch.nn as nn
from models.pact_activation import PACTActivation

def quantize_weights(w, k):
    """
    Quantification uniforme des poids dans [-1, 1].
    Approche similaire à DoReFa-Net.
    """
    if k is None or k >= 32:
        return w
    w_tanh = torch.tanh(w)
    max_abs = w_tanh.abs().max().detach().clamp(min=1e-8)
    w_norm = w_tanh / (2 * max_abs) + 0.5
    n = 2 ** k - 1
    w_scaled = w_norm * n
    w_int = torch.round(w_scaled)
    w_q = w_norm + (w_int - w_scaled).detach() / n
    return 2 * w_q - 1

class QuantConv2d(nn.Module):
    """
    Convolution 2D avec quantification des poids et activation PACT.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bit_width_w=4, bit_width_a=4, init_alpha=10.0):
        super().__init__()
        self.bit_width_w = bit_width_w
        self.bit_width_a = bit_width_a
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if bit_width_a is None:
            self.act = nn.ReLU(inplace=False)
        else:
            self.act = PACTActivation(bit_width=bit_width_a, init_alpha=init_alpha)

    def forward(self, x):
        w_q = quantize_weights(self.conv.weight, self.bit_width_w)
        x = nn.functional.conv2d(x, w_q, stride=self.conv.stride,
                                 padding=self.conv.padding, bias=self.conv.bias)
        x = self.bn(x)
        x = self.act(x)
        return x

class QuantLinear(nn.Module):
    """
    Couche fully-connected avec quantification des poids et activation PACT.
    """
    def __init__(self, in_features, out_features, bit_width_w=4, bit_width_a=None, init_alpha=10.0):
        super().__init__()
        self.bit_width_w = bit_width_w
        self.bit_width_a = bit_width_a
        self.fc = nn.Linear(in_features, out_features)
        self.act = None
        if bit_width_a is not None:
            self.act = PACTActivation(bit_width=bit_width_a, init_alpha=init_alpha)

    def forward(self, x):
        w_q = quantize_weights(self.fc.weight, self.bit_width_w)
        x = nn.functional.linear(x, w_q, self.fc.bias)
        if self.act is not None:
            x = self.act(x)
        return x
