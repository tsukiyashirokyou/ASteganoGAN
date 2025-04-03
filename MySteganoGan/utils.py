import zlib
from math import exp
import torch
from reedsolo import RSCodec
from torch.nn.functional import conv2d


rs = RSCodec(250)


def textToBits(text):
    return byteArrayToBits(textToByteArray(text))


def bitsToText(bits):
    return byteArrayToText(bitsToByteArray(bits))

#------按byte读取数据并将byte数组按bit拼接为bit数组
def byteArrayToBits(x):
    result = []
    for i in x:
        bits = bin(i)[2:]
        bits = '00000000'[len(bits):] + bits
        result.extend([int(b) for b in bits])#按位拼接bit

    return result

#------将bit数组转化成byte数组
def bitsToByteArray(bits):
    ints = []
    for b in range(len(bits) // 8):
        byte = bits[b * 8:(b + 1) * 8]
        ints.append(int(''.join([str(bit) for bit in byte]), 2))#数字bit转字符串byte
    return bytearray(ints)

#------将字符串转换为utf8-bytes，并压缩后添加纠错码
def textToByteArray(text):
    assert isinstance(text, str), "expected a string"
    x = zlib.compress(text.encode("utf-8"))
    x = rs.encode(bytearray(x))

    return x

#------将处理过的字符串流转化回字符串
def byteArrayToText(x):
    try:
        text = rs.decode(x)
        text = zlib.decompress(text)
        return text.decode("utf-8")
    except BaseException:
        return False


def first_element(storage, loc):
    """Returns the first element of two"""
    return storage

#------一维归一化高斯窗
def gaussian(windowSize, sigma):
    _exp = [exp(-(x - windowSize // 2) ** 2 / float(2 * sigma ** 2)) for x in range(windowSize)]
    gauss = torch.Tensor(_exp)
    return gauss / gauss.sum()

#------二维归一高斯窗
def create_window(windowSize, channel):
    _1DimensionWindow = gaussian(windowSize, 1.5).unsqueeze(1)
    _2DimensionWindow = _1DimensionWindow.mm(_1DimensionWindow.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2DimensionWindow.expand(channel, 1, windowSize, windowSize).contiguous()
    return window

#------ssim计算
def _ssim(img1, img2, window, windowSize, channel, sizeAverage=True):

    paddingSize = windowSize // 2 #保证卷积前后大小不变
    
    #μ
    mu1 = conv2d(img1, window, padding=paddingSize, groups=channel)
    mu2 = conv2d(img2, window, padding=paddingSize, groups=channel)
    
    mu1Sq = mu1.pow(2)
    mu2Sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    #σ
    sigma1Sq = conv2d(img1 * img1, window, padding=paddingSize, groups=channel) - mu1Sq
    sigma2Sq = conv2d(img2 * img2, window, padding=paddingSize, groups=channel) - mu2Sq
    sigma12 = conv2d(img1 * img2, window, padding=paddingSize, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    _ssimNumerator = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    _ssimDenominator = ((mu1Sq + mu2Sq + C1) * (sigma1Sq + sigma2Sq + C2))

    ssimMap = _ssimNumerator / _ssimDenominator


    if sizeAverage:
        return ssimMap.mean()#总平均值
    else:
        return ssimMap.mean(1).mean(1).mean(1)#分batch平均值


def ssim(img1, img2, windowSize=11, sizeAverage=True):
    (_, channel, _, _) = img1.size()
    window = create_window(windowSize, channel)

    device = img1.device
    window = window.to(device, dtype=img1.dtype)

    return _ssim(img1, img2, window, windowSize, channel, sizeAverage)


#------权重限制
def apply_weight_clipping(critic,limit=0.1):
    with torch.no_grad():
        for param in critic.parameters():
            param.clamp_(-limit, limit)
