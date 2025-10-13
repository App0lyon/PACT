import torch
import torch.nn as nn

class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through: d(round)/dx ≈ 1
        return grad_output

def _quantize_uniform_with_ste(x, k, alpha):
    """
    Quantification uniforme avec STE pour le gradient à travers le round.
    y_q = round(y * (2^k-1) / alpha) * alpha / (2^k-1)
    """
    if k is None or k >= 32:
        return x
    n = 2 ** k - 1
    # On suppose x déjà clampé dans [0, alpha]
    scale = n / alpha.detach()
    # round with STE
    y_int = _RoundSTE.apply(x * scale)
    y = y_int / scale
    return y

class PACTActivation(nn.Module):
    """
    Parameterized Clipping Activation (PACT)
    - y = clamp(ReLU(x), max=alpha)  (évite le mélange Number/Tensor)
    - quantification uniforme k-bits avec STE
    - alpha est un paramètre appris
    """
    def __init__(self, bit_width=4, init_alpha=10.0):
        super().__init__()
        self.bit_width = bit_width
        # alpha > 0 (option : utiliser softplus si tu veux forcer la positivité)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        # ReLU pour garantir min=0 sans passer un Number à clamp
        x_pos = torch.relu(x)
        # clamp uniquement avec max tensor => pas de mélange Number/Tensor
        x_clipped = torch.clamp(x_pos, max=self.alpha)
        if self.bit_width is not None:
            x_q = _quantize_uniform_with_ste(x_clipped, self.bit_width, self.alpha)
            return x_q
        return x_clipped

    def extra_repr(self):
        try:
            a = float(self.alpha.detach().cpu())
        except Exception:
            a = float(self.alpha.data)
        return f"bit_width={self.bit_width}, alpha={a:.4f}"
