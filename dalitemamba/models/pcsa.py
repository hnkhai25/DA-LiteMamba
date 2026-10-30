"""Pixel-wise Channel / Spatial Attention (ConvMixer-style PCA + PSA) used in the bottleneck."""
import torch
import torch.nn as nn
from einops import reduce


class PCA(nn.Module):
    """Pixel-wise channel attention."""

    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=9, groups=dim, padding="same")
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        c = reduce(x, 'b c h w -> b c', 'mean')
        x = self.dw(x)
        c_ = reduce(x, 'b c h w -> b c', 'mean')
        raise_ch = self.prob(c_ - c)
        att_score = torch.sigmoid(c_ * (1 + raise_ch))
        return torch.einsum('bchw, bc -> bchw', x, att_score)


class PSA(nn.Module):
    """Pixel-wise spatial attention."""

    def __init__(self, dim):
        super().__init__()
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)
        self.prob = nn.Softmax2d()

    def forward(self, x):
        s = reduce(x, 'b c h w -> b h w', 'mean')
        x = self.pw(x)
        s_ = reduce(x, 'b c h w -> b h w', 'mean')
        raise_sp = self.prob(s_ - s)
        att_score = torch.sigmoid(s_ * (1 + raise_sp))
        return torch.einsum('bchw, bhw -> bchw', x, att_score)


class PCSA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pca = nn.Sequential(PCA(dim), nn.GELU(), nn.BatchNorm2d(dim))
        self.psa = nn.Sequential(PSA(dim), nn.GELU(), nn.BatchNorm2d(dim))

    def forward(self, x):
        x1 = self.pca(x)
        return self.psa(x + x1)
