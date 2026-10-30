"""DA-LiteMamba: lightweight hybrid CNN-Mamba U-shaped network with dual attention."""
import torch.nn as nn
import torch.nn.functional as F

from .dacm import DACMBlock
from .pcsa import PCSA


class EncoderBlock(nn.Module):
    """DACM block -> Conv3x3/BN/ReLU (channel doubling, skip) -> MaxPool."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv33 = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU()
        self.down = nn.MaxPool2d(2, 2)
        self.mamba = DACMBlock(in_c)

    def forward(self, x):
        x = self.mamba(x)
        skip = self.act(self.bn(self.conv33(x)))
        return self.down(skip), skip


class DecoderBlock(nn.Module):
    """Upsample, gate the skip connection with a simple attention, then Conv/ReLU/BN."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.bn = nn.BatchNorm2d(in_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')

    def forward(self, x1, x2):
        x1 = self.up(x1)
        h = self.sig(self.bn(self.relu(x1 + x2)))
        return self.bn2(self.relu(self.conv(x1 + x2 * h)))


class DALiteMamba(nn.Module):
    """Returns 4 logit maps (final, and 3 deep-supervision heads), all at input resolution."""

    def __init__(self, in_channels=3, num_classes=1, widths=(16, 32, 64, 128, 256)):
        super().__init__()
        w = widths
        self.conv_in = nn.Conv2d(in_channels, w[0], kernel_size=1)

        # Encoder
        self.e1 = EncoderBlock(w[0], w[1])
        self.e2 = EncoderBlock(w[1], w[2])
        self.e3 = EncoderBlock(w[2], w[3])
        self.e4 = EncoderBlock(w[3], w[4])

        # Bottleneck
        self.b4 = PCSA(w[4])

        # Decoder
        self.d4 = DecoderBlock(w[4], w[3])
        self.d3 = DecoderBlock(w[3], w[2])
        self.d2 = DecoderBlock(w[2], w[1])
        self.d1 = DecoderBlock(w[1], w[0])

        # Map Reduce and Interpolation Blocks (MRIB) + output head
        self.conv_st = nn.Conv2d(w[3], num_classes, kernel_size=1, padding=0)
        self.conv_nd = nn.Conv2d(w[2], num_classes, kernel_size=1, padding=0)
        self.conv_rd = nn.Conv2d(w[1], num_classes, kernel_size=1, padding=0)
        self.final_layer = nn.Conv2d(w[0], num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        H, W = x.size()[2:]

        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)

        x = self.b4(x)

        x3 = self.d4(x, skip4)
        x2 = self.d3(x3, skip3)
        x1 = self.d2(x2, skip2)
        x = self.d1(x1, skip1)

        x3 = F.interpolate(self.conv_st(x3), (H, W), mode='bilinear', align_corners=False)
        x2 = F.interpolate(self.conv_nd(x2), (H, W), mode='bilinear', align_corners=False)
        x1 = F.interpolate(self.conv_rd(x1), (H, W), mode='bilinear', align_corners=False)
        x = self.final_layer(x)
        return x, x1, x2, x3
