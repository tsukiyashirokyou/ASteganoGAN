from MySteganoGan.activation_funtion import ACON_C
from MySteganoGan.attention import *
import torch.nn as nn
from torch.nn.utils import spectral_norm
from MySteganoGan.residual_block import ResidualBlock

class BasicCritic(nn.Module):
#------卷积模块
    def _conv2d(self, inChannels, outChannels):
        return nn.Conv2d(
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=3,
            padding=1,
        )

#------卷积层模块
    def _conv2dBlock(self,inChannels=3):
        return nn.Sequential(
            spectral_norm(self._conv2d(inChannels, self.hiddenSize)),
            ACON_C(self.hiddenSize),
            nn.BatchNorm2d(self.hiddenSize),
            AdaptiveAttention(self.hiddenSize),
        )

#------残差网络模块
    def _residual_block(self, inChannels, outChannels):
        return ResidualBlock(inChannels,outChannels)
#------构建模型
    def _build_models(self):
        return nn.Sequential(
            # self._conv2dBlock(),
            # self._conv2dBlock(self.hiddenSize),
            # self._conv2dBlock(self.hiddenSize),
            self._residual_block(3,self.hiddenSize),
            self._residual_block(self.hiddenSize,self.hiddenSize),
            self._residual_block(self.hiddenSize,self.hiddenSize),
            # spectral_norm(self._conv2d(self.hiddenSize, 1))
            self._conv2d(self.hiddenSize, 1)
        )
#------初始化s
    def __init__(self, hiddenSize):
        super().__init__()
        self.hiddenSize = hiddenSize
        self._models = self._build_models()
#------前向传播
    def forward(self, x):
        x = self._models(x)
        x = torch.mean(x.view(x.size(0), -1), dim=1)#得分张量，仿自适应均质池化操作

        return x
