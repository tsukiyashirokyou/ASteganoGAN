from MySteganoGan import train
from pathlib import  Path
import torch
from MySteganoGan.utils import *
from MySteganoGan.models import SteganoGAN


def main():
    model_list = ['ResidualEncoder','BasicEncoder','DenseEncoder']
    dataDepth_list = [5,4,2]
    tran_list = [
        {'model': x ,'dataDepth': y}
        for x in model_list
        for y in dataDepth_list
    ]
    # tran_list = [
    #     {
    #         'model':'DenseEncoder',
    #         'dataDepth':1
    #     },
    #     {
    #         'model':'DenseEncoder',
    #         'dataDepth':3
    #     },
    # ]

    for i in range(len(tran_list)):
        train.trainSteganoGAN(**tran_list[i])

    # model_path = Path(r'/root/pythonProject/MyGraduationProject/logs/models/DenseEncoder6_1745587648/bestModel.pth')
    # test_picture_path = Path(r'/root/pythonProject/MyGraduationProject/test/test_picture.png')
    # output_path = Path(r'/root/pythonProject/MyGraduationProject/test/test_encoded.png')
    #
    # ASteganoGAN = SteganoGAN.load(model_path)
    # ASteganoGAN.encode(coverImagePath=test_picture_path,outputImagePath=output_path,text="这是测试用解密信息")
    # print(ASteganoGAN.decode(output_path))




if __name__ == '__main__':
    main()
