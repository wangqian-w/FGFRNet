from model.utils import *

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Frequency and gradient feature refinement for infrared small target detection')
    # choose model
    parser.add_argument('--model', type=str, default='FGFRNet',
                        help='model name: FGFRNet')

    # parameter for DNANet
    parser.add_argument('--channel_size', type=str, default='three',
                        help='one,  two,  three,  four')

    # data and pre-process
    parser.add_argument('--dataset', type=str, default='NUAA-SIRST',
                        help='dataset name: NUAA-SIRST, NUDT-SIRST, IRSTD-1k')
    parser.add_argument('--st_model', type=str, default='NUAA-SIRST_FGFRNet_22_05_2025_14_35_34_woDS',
                        help='NUDT-SIRST_FGFRNet_24_04_2025_16_59_03_woDS, IRSTD-1k_FGFRNet_24_04_2025_16_59_03_woDS')
    parser.add_argument('--model_dir', type=str,
                        default = 'NUAA-SIRST_FGFRNet_22_05_2025_14_35_34_woDS/FGFRNet_NUAA-SIRST_best_miou.pth.tar',
                        help    = 'NUDT-SIRST_FGFRNet_24_04_2025_16_59_03_woDS/FGFRNet_NUDT-SIRST_best_miou.pth.tar,'
                                  'IRSTD-1k_FGFRNet_24_04_2025_16_59_03_woDS/FGFRNet_IRSTD-1k_best_miou.pth.tar')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.2', help='when --mode==Ratio')
    parser.add_argument('--root', type=str, default='dataset/')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='8_2',
                        help='8_2')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='base image size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='crop image size')

    #  hyper params for training
    parser.add_argument('--epochs', type=int, default=1500, metavar='N',
                        help='number of epochs to train (default: 110)')
    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    # ROC threshold
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='crop image size')

    args = parser.parse_args()

    # the parser
    return args