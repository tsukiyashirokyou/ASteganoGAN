import torch.nn as nn
from MySteganoGan.activation_funtion import *
from MySteganoGan.attention import *
from torch.nn.utils import spectral_norm
from torch.nn.functional import celu

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            ACON_C(out_channels),
            nn.BatchNorm2d(out_channels),
            AdaptiveAttention(out_channels)
        )
        # 如果通道数不匹配，用1x1卷积调整
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.conv_block(x)
