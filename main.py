from MySteganoGan import train
from pathlib import  Path
import torch
def main():
    # model_list = ['DenseEncoder','ResidualEncoder','BasicEncoder']
    # dataDepth_list = [1,2,3,4,5,6]
    # tran_list = [
    #     {'model': x ,'dataDepth': y}
    #     for x in model_list
    #     for y in dataDepth_list
    # ]
    tran_list = [
        {
            'model':'DenseEncoder',
            'dataDepth':5
        },
        {
            'model': 'BasicEncoder',
            'dataDepth': 6
        }
    ]

    for i in range(len(tran_list)):
        train.trainSteganoGAN(**tran_list[i])

if __name__ == '__main__':
    main()
