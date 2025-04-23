import gc
import inspect
import json
from collections import Counter
from  pathlib import Path
import imageio
import torch
from imageio import  imwrite
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from PIL import Image
import numpy as np
from lion_pytorch import Lion
from MySteganoGan.utils import apply_weight_clipping

from MySteganoGan.utils import bitsToByteArray, byteArrayToText, ssim, textToBits

DEFAULT_PATH = Path(__file__).resolve().parent / 'train'

METRIC_FIELDS = [
    'val.encoderMseLoss',
    'val.decoderCrossEntropyLoss',
    'val.decoderAcc',
    'val.coverImagesScore',
    'val.generatedScore',
    'val.ssim',
    'val.psnr',
    'val.bpp',
    'train.encoderMseLoss',
    'train.decoderCrossEntropyLoss',
    'train.decoderAcc',
    'train.coverImagesScore',
    'train.generatedScore',
]


class SteganoGAN(object):
#------统一传参生成实例
    def _getInstance(self, classOrInstance, kwargs):

        if not inspect.isclass(classOrInstance):
            return classOrInstance

        argspec = inspect.getfullargspec(classOrInstance.__init__).args
        argspec.remove('self')
        initArgs = {arg: kwargs[arg] for arg in argspec}

        return classOrInstance(**initArgs)
#------设备设置
    def setDevice(self, cuda=True,details = True):
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
            if details:
                print('GPU Mode')
        else:
            self.cuda = False
            self.device = torch.device('cpu')
            if details:
                print('CPU Mode')

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.critic.to(self.device)

#------初始化
    def __init__(self, dataDepth, encoder, decoder, critic,
                 cuda=False, modelArgsDic=None, logDir=None, **kwargs):

        self.dataDepth = dataDepth
        kwargs['dataDepth'] = dataDepth
        self.encoder = self._getInstance(encoder, kwargs)
        self.decoder = self._getInstance(decoder, kwargs)
        self.critic = self._getInstance(critic, kwargs)
        self.setDevice(cuda)

        self.criticOptimizer = None
        self.decoderOptimizer = None
        self.modelArgsDic = None


        # Misc
        self.fitMetrics = None
        self.history = list()

    #------日志目录
        self.logDir = Path(logDir) if logDir else None
        if self.logDir:
            self.logDir.mkdir(parents=True, exist_ok=True)
            self.samplesPath = self.logDir / 'samples'
            self.samplesPath.mkdir(parents=True, exist_ok=True)

#------生成随机密文张量
    def _randomData(self, coverImages):
        N, _, H, W = coverImages.size()
        return torch.randint(0,2,(N, self.dataDepth, H, W), device=self.device).float()
#------训练用加解密
    def _encodeAndDecode(self, coverImages, quantize=False):
        payload = self._randomData(coverImages)
        generated = self.encoder(coverImages, payload)

        if quantize:
            generated = (255.0 * (generated + 1.0) / 2.0).long()
            generated = 2.0 * generated.float() / 255.0 - 1.0

        decoded = self.decoder(generated)

        return generated, payload, decoded

#------batch critic得分
    def _critic(self, image):
        return torch.mean(self.critic(image))

#------设置优化器
    def _getOptimizers(self):
        self.__init__lr = 8e-5
        _encDecParamList = list(self.decoder.parameters()) + list(self.encoder.parameters())
        criticOptimizer = Lion(self.critic.parameters(), lr=self.__init__lr,weight_decay=0.03)
        decoderOptimizer = Lion(_encDecParamList, lr=self.__init__lr,weight_decay=0.03)
        # criticOptimizer = AdamW(self.critic.parameters(), lr=self.__init__lr)
        # decoderOptimizer = AdamW(_encDecParamList, lr=self.__init__lr)
        self.update_fn = criticOptimizer.update_fn
        return criticOptimizer, decoderOptimizer

#------训练评估器
    def _fitCritic(self, train, metrics):
        for coverImages, _ in tqdm(train):
            gc.collect()
            coverImages = coverImages.to(self.device)
            payload = self._randomData(coverImages)
            generated = self.encoder(coverImages, payload)
            coverImagesScore = self._critic(coverImages)
            generatedScore = self._critic(generated)

            self.criticOptimizer.zero_grad()

            (coverImagesScore - generatedScore).backward(retain_graph=False)
            clip_grad_norm_(self.critic.parameters(), max_norm=0.25)#梯度裁剪
            self.criticOptimizer.step()

            apply_weight_clipping(self.critic)#权重限制


            metrics['train.coverImagesScore'].append(coverImagesScore.item())
            metrics['train.generatedScore'].append(generatedScore.item())

#------训练编码器和解码器
    def _fitEncAndDec(self, train, metrics):
        for coverImages, _ in tqdm(train):
            gc.collect()
            coverImages = coverImages.to(self.device)
            generated, payload, decoded = self._encodeAndDecode(coverImages)
            encoderMseLoss, decoderCrossEntropyLoss, decoderAcc= self._encDecScores(
                coverImages, generated, payload, decoded)
            generatedScore = self._critic(generated)

            self.decoderOptimizer.zero_grad()
            (100.0 * encoderMseLoss + decoderCrossEntropyLoss + generatedScore).backward()
            clip_grad_norm_(self.critic.parameters(), max_norm=0.25)#梯度裁剪
            self.decoderOptimizer.step()

            metrics['train.encoderMseLoss'].append(encoderMseLoss.item())
            metrics['train.decoderCrossEntropyLoss'].append(decoderCrossEntropyLoss.item())
            metrics['train.decoderAcc'].append(decoderAcc.item())

#------编码器解码器得分计算
    def _encDecScores(self, coverImages, generated, payload, decoded):
        encoderMseLoss = mse_loss(generated, coverImages)
        decoderCrossEntropyLoss = binary_cross_entropy_with_logits(decoded, payload)
        decoderAcc = (decoded >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel()

        return encoderMseLoss, decoderCrossEntropyLoss, decoderAcc

#------验证集验证
    def _validate(self, validate, metrics):
        for coverImages, _ in tqdm(validate):
            gc.collect()
            coverImages = coverImages.to(self.device)
            generated, payload, decoded = self._encodeAndDecode(coverImages, quantize=True)
            encoderMseLoss, decoderCrossEntropyLoss, decoderAcc= self._encDecScores(
                coverImages, generated, payload, decoded)
            generatedScore = self._critic(generated)
            coverImagesScore = self._critic(coverImages)


            metrics['val.encoderMseLoss'].append(encoderMseLoss.item())
            metrics['val.decoderCrossEntropyLoss'].append(decoderCrossEntropyLoss.item())
            metrics['val.decoderAcc'].append(decoderAcc.item())
            metrics['val.coverImagesScore'].append(coverImagesScore.item())
            metrics['val.generatedScore'].append(generatedScore.item())
            metrics['val.ssim'].append(ssim(coverImages, generated).item())
            # psnr=20log10(sc)-10log10(mse)->10log10(sc^2/mse),sc为像素最大差值,(原[0,1]归一化为[-1,1])
            metrics['val.psnr'].append(10 * torch.log10(4 / encoderMseLoss).item())
            #bpp=D*(1-2p),p为错误率->bpp=D*(2ACC-1)
            metrics['val.bpp'].append(self.dataDepth * (2 * decoderAcc.item() - 1))
#------生成样例
    def _generateSamples(self, samplesPath, coverImages, epoch):
        coverImages = coverImages.to(self.device)
        generated, payload, decoded = self._encodeAndDecode(coverImages)
        samples = generated.size(0)
        print('samples: {}'.format(samples))
        print('coverImages: {}'.format(coverImages.size(0)))
        for sample in range(samples):
            coverImagesPath = samplesPath / '{}.coverImages.png'.format(sample)
            sampleName = '{}.generated-{:2d}.png'.format(sample, epoch)
            samplePath = samplesPath / sampleName

            image = (coverImages[sample].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2.0
            imageio.imwrite(coverImagesPath, (255.0 * image).astype('uint8'))

            sampled = generated[sample].clamp(-1.0, 1.0).permute(1, 2, 0)
            sampled = sampled.detach().cpu().numpy() + 1.0

            image = sampled / 2.0
            imageio.imwrite(samplePath, (255.0 * image).astype('uint8'))

#------比较最好模型
    def bestModelEvalution(self,metricsBest):
        return (metricsBest['val.ssim'] * 2 + metricsBest['val.bpp'] <
                self.fitMetrics['val.ssim'] * 2 + self.fitMetrics['val.bpp'])
#------SteganoGan训练
    def fit(self, train, validate, epochs=5):

        #------尝试载入过去最好的模型
        bestModelExists = False
        try:
            bestModel = self.load(path=self.logDir,modelArgsDic=self.modelArgsDic,details=False)
            bestModelExists = True
        except ValueError:
            pass


        if self.criticOptimizer is None:
            self.criticOptimizer, self.decoderOptimizer= self._getOptimizers()
            self.epochs = 0

        if self.logDir:
            sampleCoverImages = next(iter(validate))[0]

        self.total = self.epochs + epochs #total为总训练轮数，epochs为本次训练需要训练的轮数

        flag = False
        for epoch in range(1, epochs + 1):
            self.epochs += 1 #已训练轮数

            metrics = {metric: list() for metric in METRIC_FIELDS}

            print('Epoch {}/{}'.format(self.epochs, self.total))

            self._fitCritic(train, metrics)
            self._fitEncAndDec(train, metrics)
            self._validate(validate, metrics)

            self.fitMetrics = {k: sum(v) / len(v) for k, v in metrics.items()}#各batch平均指标
            self.fitMetrics['epoch'] = epoch

            print('bpp-{:0.3f},ssim-{:0.3f}'.format(self.fitMetrics['val.bpp'],
                                                  self.fitMetrics['val.ssim']))

            if self.logDir:
                self.history.append(self.fitMetrics)
            #------保存样例
                metricsPath = self.logDir / 'metrics.{}.log'.format(epoch)
                with metricsPath.open('w') as metricsFile:
                    json.dump(self.history, metricsFile, indent=4)

                saveModelName = '{}.bpp-{:03f}.pth'.format(
                    self.epochs, self.fitMetrics['val.bpp'])
            #------保存模型
                self.save(self.logDir / saveModelName)
                #------新建最好模型或最好模型更新

                if (not bestModelExists) or self.bestModelEvalution(metricsBest=bestModel.history[-1]):
                    if flag:
                        print(not bestModelExists, self.bestModelEvalution(metricsBest=bestModel.history[-1]))
                    else:
                        flag = True

                    self.save(self.logDir / 'bestModel.pth')#保存当前模型
                    bestModel = self.load(path=self.logDir,modelArgsDic=self.modelArgsDic,details=False)

                    with (self.logDir / 'bestMetrics.log').open('w') as bestMetricsFile:
                        json.dump(bestModel.history[-1], bestMetricsFile, indent=4)

                    bestModelExists = True

            #------生成样例
                self._generateSamples(self.samplesPath, sampleCoverImages, epoch)

            if self.cuda:
                torch.cuda.empty_cache()

            gc.collect()


#------明文载荷，明文转四维张量
    def _makePayload(self, width, height, depth, text):

        message = textToBits(text) + [0] * 32

        payload = message
        while len(payload) < width * height * depth:
            payload += message

        payload = payload[:width * height * depth]

        return torch.FloatTensor(payload).view(1, depth, height, width)
#------编码
    def encode(self, coverImagePath, outputImagePath, text):

        coverImage = np.array(Image.open(coverImagePath).convert('RGB')) / 127.5 - 1.0
        coverImage = torch.FloatTensor(coverImage).permute(2, 1, 0).unsqueeze(0)

        coverImageSize = coverImage.size()
        payload = self._makePayload(coverImageSize[3], coverImageSize[2], self.dataDepth, text)

        coverImage = coverImage.to(self.device)
        payload = payload.to(self.device)
        generated = self.encoder(coverImage, payload)[0].clamp(-1.0, 1.0)

        generated = (generated.permute(2, 1, 0).detach().cpu().numpy() + 1.0) * 127.5
        imwrite(outputImagePath, generated.astype('uint8'))

        print('Encoding completed.')
#------解码
    def decode(self, imagePath):

        if not imagePath.exists():
            raise ValueError('Unable to read %s.' % imagePath)

        image = np.array(Image.open(imagePath).convert('RGB')) / 255
        image = torch.FloatTensor(image).permute(2, 1, 0).unsqueeze(0)
        image = image.to(self.device)

        image = self.decoder(image).view(-1) > 0

        # split and decode messages
        candidates = Counter()
        bits = image.detach().cpu().numpy().tolist()
        for candidate in bitsToByteArray(bits).split(b'\x00\x00\x00\x00'):
            candidate = byteArrayToText(bytearray(candidate))
            if candidate:
                candidates[candidate] += 1

        if len(candidates) == 0:
            raise ValueError('Failed to find message.')

        candidate, count = candidates.most_common(1)[0]
        return candidate

#------保存模型
    def save(self, path):
        torch.save(self, path)

#------优化器张量转移
    def optimizerTo(self,optimizer, device):
        try:
            optimizer.update_fn = self.update_fn
            optimizer.decoupled_wd = False
            optimizer._init_lr =self.__init__lr
        except AttributeError:
            pass
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def reloadArgs(self,modelArgsDic):

        self.logDir = modelArgsDic['logDir']
        self.samplesPath = self.logDir / 'samples'
        self.hiddenSize = modelArgsDic['hiddenSize']
        self.modelArgsDic = modelArgsDic

        self.samplesPath.mkdir(parents=True, exist_ok=True)

        return self

    #------载入模型
    @classmethod
    def load(cls, path=None, cuda=True ,details = True ,modelArgsDic = None):

        if path == None:
            path = Path(__file__).parent.parent / 'bestModel.pth'
        else:
            path = Path(path) / 'bestModel.pth'
        if path.exists():
            steganogan = torch.load(path, map_location='cpu',weights_only=False)

            if modelArgsDic:
                steganogan = steganogan.reloadArgs(modelArgsDic)

            steganogan.setDevice(cuda,details=details)
            steganogan.optimizerTo(steganogan.criticOptimizer,steganogan.device)
            steganogan.optimizerTo(steganogan.decoderOptimizer,steganogan.device)
            return steganogan
        else:
            raise ValueError('path not found.')

        return steganogan
