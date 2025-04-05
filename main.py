from MySteganoGan import train
from pathlib import  Path
import torch
def main():
    # tran_list = [
    #     {
    #         'model':'DenseEncoder',
    #         'dataDepth':6
    #     },
    #     {
    #         'model':'ResidualEncoder',
    #         'dataDepth':6
    #     },
    #     {
    #         'model':'BasicEncoder',
    #         'dataDepth':6
    #     }
    # ]
    tran_list = [
        {
            # 'model':'ResidualEncoder',
            'model':'DenseEncoder',
            'dataDepth':6
        }
    ]
    for i in range(len(tran_list)):
        train.trainSteganoGAN(**tran_list[i])

if __name__ == '__main__':
    main()
