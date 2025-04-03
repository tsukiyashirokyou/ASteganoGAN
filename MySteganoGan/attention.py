import torch.nn as nn
from MySteganoGan.activation_funtion import *

class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_fc = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, kernel_size=1, bias=False),
            ACON_C(channel // ratio),
            nn.Conv2d(channel // ratio, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel)  # 增加 BatchNorm 提高数值稳定性
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局池化特征
        avg_out = self.shared_fc(self.avg_pool(x))
        max_out = self.shared_fc(self.max_pool(x))
        # 合并注意力权重并加权
        out = self.sigmoid(avg_out + max_out)
        high_freq = x - torch.mean(x, dim=(2, 3), keepdim=True)
        return x * out + high_freq * 0.1  # 加权融合

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class AdaptiveAttention(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel_att = ChannelAttention(channel)
        self.spatial_att = SpatialAttention()
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 可学习权重参数
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.gate = nn.Sequential(
            nn.Conv2d(channel, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_out = self.channel_att(x)
        spatial_out = self.spatial_att(x)
        gate_weight = self.gate(x)
        return x + gate_weight * (self.alpha * channel_out + self.beta * spatial_out)
