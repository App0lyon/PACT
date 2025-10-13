import torch
import torch.nn as nn

def quantize_uniform(x, k, alpha):
    """
    Quantification uniforme (PACT Eq. 2)
    """
    n = 2 ** k - 1
    scale = n / alpha
    x_clipped = torch.clamp(x, 0, alpha)
    x_q = torch.round(x_clipped * scale) / scale
    return x_q

def l2_regularization(model):
    """
    Calcule la régularisation L2 sur les poids.
    """
    l2 = 0.0
    for p in model.parameters():
        if p.requires_grad:
            l2 += torch.sum(p.pow(2))
    return l2

def alpha_regularization(model):
    """
    Calcule la régularisation L2 uniquement sur les paramètres alpha (PACT).
    """
    reg = 0.0
    for name, param in model.named_parameters():
        if "alpha" in name:
            reg += torch.sum(param.pow(2))
    return reg
