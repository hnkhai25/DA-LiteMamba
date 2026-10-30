"""Double Attention CNN and Mamba (DACM) block — Sec. 3.1 of the paper."""
import torch
import torch.nn as nn

from .vss import VSSBlock


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    # (B, C, H, W) -> (B, groups, C // groups, H, W) -> transpose -> flatten
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, num_channels, height, width)


class DoubleAttentionGate(nn.Module):
    """Double Attention Gate (DAGB): global and local features re-weight each other,
    then are concatenated and channel-shuffled."""

    def __init__(self, in_c):
        super().__init__()
        self.in_c = in_c
        # Shared depthwise conv for both branches.
        self.DWConv = nn.Conv2d(in_c // 2, in_c // 2, kernel_size=3, groups=in_c // 2, padding=1)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        y1 = self.sigmoid(self.relu(self.DWConv(x1)))
        y2 = self.sigmoid(self.gelu(self.DWConv(x2)))
        output = torch.cat((x1 * y2, x2 * y1), dim=1)
        return channel_shuffle(output, self.in_c // 2)


class DACMBlock(nn.Module):
    """Channel split -> (VSS block | DW/PW conv block) -> Double Attention Gate."""

    def __init__(self, in_c):
        super().__init__()
        self.mamba = VSSBlock(in_c // 2)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_c // 2),

            nn.Conv2d(in_c // 2, in_c // 2, kernel_size=3, padding=1, groups=in_c // 2),
            nn.BatchNorm2d(in_c // 2),
            nn.ReLU(),

            nn.Conv2d(in_c // 2, in_c, kernel_size=3, padding=1, groups=in_c // 2),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),

            nn.Conv2d(in_c, in_c // 2, kernel_size=1),
            nn.BatchNorm2d(in_c // 2),
            nn.ReLU()
        )
        # Attribute name kept as `block` so original checkpoints load unchanged.
        self.block = DoubleAttentionGate(in_c)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.mamba(x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x2 = self.conv(x2)
        return self.block(x1, x2)
