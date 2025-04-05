from MySteganoGan import train
from pathlib import  Path
import torch
def main():
    # model_list = ['DenseEncoder','ResidualEncoder','BasicEncoder']
    model_list = ['ResidualEncoder','BasicEncoder']
    dataDepth_list = [4,5]
    tran_list = [
        {'model': x ,'dataDepth': y}
        for x in model_list
        for y in dataDepth_list
    ]

    for i in range(len(tran_list)):
        train.trainSteganoGAN(**tran_list[i])

if __name__ == '__main__':
    main()
