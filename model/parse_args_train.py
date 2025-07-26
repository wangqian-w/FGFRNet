from model.utils import *

def parse_args():
    """Training ptions for Segmentation Experiments"""
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
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.2', help='when mode==Ratio')
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
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--train_batch_size', type=int, default=8,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test_batch_size', type=int, default=8,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--min_lr', default=1e-5,
                        type=float, help='minimum learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD (default: 0.9)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='0.05,learning rate (default: 0.1)')
    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    args = parser.parse_args()
    # make dir for save result
    args.save_dir = make_dir( args.dataset, args.model)
    # save training log
    save_train_log(args, args.save_dir)
    # the parser
    return args
