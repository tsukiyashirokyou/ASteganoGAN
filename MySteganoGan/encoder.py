from torch.nn.utils.parametrizations import spectral_norm
from MySteganoGan.attention import *
from MySteganoGan.activation_funtion import *
import torch.nn as nn
from MySteganoGan.residual_block import ResidualBlock

class BasicEncoder(nn.Module):

    addImage = False
#------卷积模块
    def _conv2d(self, inChannels, outChannels):
        return nn.Conv2d(
            in_channels=inChannels,
            out_channels=outChannels,
            kernel_size=3,
            padding=1
        )
#------卷积层模块
    def _conv2dBlock(self,inChannels=3):
        return nn.Sequential(
            self._conv2d(inChannels, self.hiddenSize),
            ACON_C(self.hiddenSize),
            nn.BatchNorm2d(self.hiddenSize),
            AdaptiveAttention(self.hiddenSize),
        )

# ------残差网络模块
    def _residual_block(self, inChannels,outChannels):
        return ResidualBlock(inChannels,outChannels)
#------构建模型
    def _buildModels(self):
        self.features = self._conv2dBlock()#载体图像特征
        self.layers = nn.Sequential(
            self._residual_block(self.hiddenSize+self.dataDepth,self.hiddenSize),
            self._residual_block(self.hiddenSize,self.hiddenSize),
            self._residual_block(self.hiddenSize, self.hiddenSize),
            self._conv2d(self.hiddenSize, 3),
        )
        return [self.features, self.layers]
#------初始化
    def __init__(self, dataDepth, hiddenSize):
        super().__init__()
        self.dataDepth = dataDepth
        self.hiddenSize = hiddenSize
        self._models = self._buildModels()
#------前向传播
    def forward(self, image, data):
        x = self._models[0](image)
        xList = [x]
    #------复用特征
        for layer in self._models[1:]:
            x = layer(torch.cat(xList + [data], dim=1))
            xList.append(x)

        if self.addImage:
            x = (image + x)
        return x

class ResidualEncoder(BasicEncoder):
    addImage = True
    # ------去除了tanh，并且启用了加算载体特征
    def _buildModels(self):
        self.features = self._conv2dBlock()
        self.layers = nn.Sequential(
            self._residual_block(self.hiddenSize+self.dataDepth,self.hiddenSize),
            self._residual_block(self.hiddenSize,self.hiddenSize),
            self._residual_block(self.hiddenSize,self.hiddenSize),
            self._conv2d(self.hiddenSize, 3),
        )
        return self.features, self.layers


class DenseEncoder(BasicEncoder):
    addImage = True

    def _buildModels(self):
        self.conv1 = self._conv2dBlock()
        self.conv2 = self._conv2dBlock(self.hiddenSize + self.dataDepth)
        self.conv3 = self._conv2dBlock(self.hiddenSize * 2 + self.dataDepth)
        self.conv4 = self._conv2dBlock(self.hiddenSize * 3 + self.dataDepth)
        self.conv5 = nn.Sequential(
            self._conv2d(self.hiddenSize*4+self.dataDepth, 3)
        )

        return [self.conv1, self.conv2, self.conv3, self.conv4,self.conv5]
