import os
import argparse


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu-flag', action='store_true',
                        help='flag for using gpu', default=True)

    parser.add_argument('--train-flag', action='store_true',
                        help='flag for training network', default=False)#False

    parser.add_argument('--hard-prune-flag', action='store_true',
                        help='flag for pruning network', default=True)#True

    parser.add_argument('--soft-prune-flag', action='store_true',
                        help='flag for soft pruning network', default=False)

    parser.add_argument('--test-flag', action='store_true',
                        help='flag for testing network', default=False)

    parser.add_argument('--network', type=str,
                        help='Network for training', default='vgg')

    parser.add_argument('--data-set', type=str,
                        help='Data set for training network', default='CIFAR10')

    parser.add_argument('--data-path', type=str,
                        help='Path of dataset', default='./data')

    parser.add_argument('--epoch', type=int, 
                        help='number of epoch for training network', default=20)

    parser.add_argument('--batch-size', type=int,
                        help='batch size', default=128)

    parser.add_argument('--lr', type=float, 
                        help='learning rate', default=0.001)

    # ref: https://github.com/kuangliu/pytorch-cifar
    parser.add_argument('--lr-milestone', nargs='+', type=int, 
                        help='list of epoch for adjust learning rate', default= [100, 200])

    parser.add_argument('--lr-gamma', type=float,
                        help='factor for decay learning rate', default=0.1)

    parser.add_argument('--momentum', type=float,
                        help='momentum for optimizer', default=0.9)

    parser.add_argument('--weight-decay', type=float,
                        help='factor for weight decay in optimizer', default=5e-4)

    parser.add_argument('--imsize', type=int,
                        help='size for image resize', default=None)

    parser.add_argument('--cropsize', type=int,
                        help='size for image crop', default=32)

    parser.add_argument('--crop-padding', type=int,
                        help='size for padding in image crop', default=4)

    parser.add_argument('--hflip', type=float,
                        help='probability of random horizontal flip', default=0.5)

    parser.add_argument('--load-path', type=str,
                        help='trained model load path to prune', default='./trained_models/vgg.pth')#./prunned_models/vgg.pth,./trained_models/vgg.pth

    parser.add_argument('--save-path', type=str,
                        help='model save path', default='./prunned_models/vgg.pth')#required=True,./trained_prunned_models/vgg.pth,./trained_models/vgg.pth

    parser.add_argument('--independent-prune-flag', action='store_true',
                        help='prune multiple layers by "independent strategy"', default=False)

    parser.add_argument('--prune-layers', nargs='+',
                        help='layer index for pruning', default=['conv1', 'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13'])#['conv1', 'conv8', 'conv9', 'conv10', 'conv11', 'conv12', 'conv13']

    parser.add_argument('--prune-channels', nargs='+', type=int,
                        help='number of channel to prune layers', default=[32, 256 ,256, 256, 256, 256, 256])#[32, 256 ,256, 256, 256, 256, 256]

    parser.add_argument('--prune-rate', nargs='+', type=float,
                        help='factor for soft filter pruning', default=[0.125, 0.125, 0.125])

    return parser


def get_parameter():
    parser = build_parser()
    args = parser.parse_args()

    print("-*-" * 10 + "\n\t\tArguments\n" + "-*-" * 10)
    for key, value in vars(args).items():
        print("%s: %s" % (key, value))

    save_folder = args.save_path[0:args.save_path.rindex('/')]
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print("Make dir: ", save_folder)

    return args
