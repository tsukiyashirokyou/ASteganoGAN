from MySteganoGan.attention import *
from MySteganoGan.activation_funtion import *
import torch.nn as nn
from MySteganoGan.residual_block import ResidualBlock
from torch.nn.functional import celu
from torch.nn.utils import spectral_norm

class BasicDecoder(nn.Module):
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
            self._conv2d(inChannels, self.hiddenSize),
            ACON_C(self.hiddenSize),
            nn.BatchNorm2d(self.hiddenSize),
            AdaptiveAttention(self.hiddenSize),

        )
    def _residual_block(self, inChannels, outChannels):
        return ResidualBlock(inChannels,outChannels)
#------构建模型
    def _buildModels(self):
        self.layers = nn.Sequential(
            # self._conv2dBlock(),
            # self._conv2dBlock(self.hiddenSize),
            # self._conv2dBlock(self.hiddenSize),
            # self._conv2dBlock(self.hiddenSize),
            self._residual_block(3,self.hiddenSize),
            self._residual_block(self.hiddenSize,self.hiddenSize),
            self._residual_block(self.hiddenSize,self.hiddenSize),
            self._residual_block(self.hiddenSize, self.hiddenSize),
            self._conv2d(self.hiddenSize, self.dataDepth)
        )

        return [self.layers]
#------初始化
    def __init__(self, dataDepth, hiddenSize):
        super().__init__()
        self.dataDepth = dataDepth
        self.hiddenSize = hiddenSize
        self._models = self._buildModels()
#------前向传播
    def forward(self, x):
        x = self._models[0](x)

        if len(self._models) > 1:
            x_list = [x]
            for layer in self._models[1:]:
                x = layer(torch.cat(x_list, dim=1))
                x_list.append(x)

        return x

class DenseDecoder(BasicDecoder):

    def _buildModels(self):
        self.conv1 = self._conv2dBlock()
        self.conv2 = self._conv2dBlock(self.hiddenSize)
        self.conv3 = self._conv2dBlock(self.hiddenSize*2)
        self.conv4 = self._conv2dBlock(self.hiddenSize*3)
        # self.conv1 = self._residual_block(3,self.hiddenSize)
        # self.conv2 = self._residual_block(self.hiddenSize,self.hiddenSize)
        # self.conv3 = self._residual_block(self.hiddenSize*2,self.hiddenSize)
        # self.conv4 = self._residual_block(self.hiddenSize*3,self.hiddenSize)
        self.conv5 = nn.Sequential(
            self._conv2d(self.hiddenSize * 3, self.dataDepth)
        )
        return [self.conv1, self.conv2, self.conv3, self.conv4,self.conv5]

