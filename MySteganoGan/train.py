import json
from time import time
from pathlib import Path
import torch

from MySteganoGan.models import SteganoGAN
from MySteganoGan.critic import BasicCritic
from MySteganoGan.decoder import DenseDecoder
from MySteganoGan.encoder import BasicEncoder, DenseEncoder, ResidualEncoder
from MySteganoGan.myDataLoader import DataLoader


def trainSteganoGAN(model='DenseEncoder',dataDepth=6):

    model_dict = {
        'DenseEncoder': DenseEncoder,
        'ResidualEncoder': ResidualEncoder,
        'BasicEncoder': BasicEncoder,
    }

    #------基本参数
    torch.manual_seed(42)
    timestamp = int(time())

    dataPath = Path(__file__).resolve().parent.parent / 'data'
    logDir = Path(__file__).resolve().parent.parent / 'logs' / "models" / (model+str(dataDepth)+"_"+str(timestamp))
    jsonDir = logDir / "config.json"
    weightsDir = logDir / "weights.steg"


    dataDepth = dataDepth
    encoder = model_dict[model]
    hiddenSize = 32
    epochs = 32

    modelArgsDic = {
        'dataDepth':dataDepth,
        'encoder':encoder,
        'decoder':DenseDecoder,
        'critic':BasicCritic,
        'hiddenSize':hiddenSize,
        'cuda':True,
        'logDir':logDir,
        'epochs':epochs,
    }

    #------创建目录
    logDir.mkdir(parents=True, exist_ok=True)

    #------数据集
    train = DataLoader(dataPath / 'train', shuffle=True)
    validation = DataLoader(dataPath / 'val', shuffle=False)

    #------模型
    try:
        steganogan = SteganoGAN.load(path=logDir,modelArgsDic=modelArgsDic)
    except ValueError:
        steganogan = SteganoGAN(modelArgsDic=modelArgsDic,**modelArgsDic)

    with jsonDir.open('wt') as fout:
        fout.write(json.dumps(modelArgsDic, indent=2, default=lambda o: str(o)))

    print('现在训练的模型是:'+model+str(dataDepth))
    steganogan.fit(train, validation, epochs=modelArgsDic['epochs'])

    steganogan.save(weightsDir)

