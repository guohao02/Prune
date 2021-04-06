import os
import argparse
import models
import torch

def build_parser():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')#/home/guohao/gh/PruningFilters/data/cifar-10-batches-py/
    # parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Choose between Cifar10/100 and ImageNet.')
    parser.add_argument('--arch', metavar='ARCH', default='resnet20', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnext29_8_64)')
    # Optimization options
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='The Learning Rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
    # Checkpoints
    parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
    parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    # Acceleration
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
    # random seed
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    #compress rate
    parser.add_argument('--rate', type=float, default=0.9, help='compress rate of model')
    parser.add_argument('--layer_begin', type=int, default=3,  help='compress layer of model')
    parser.add_argument('--layer_end', type=int, default=57,  help='compress layer of model')
    parser.add_argument('--layer_inter', type=int, default=3,  help='compress layer of model')
    parser.add_argument('--epoch_prune', type=int, default=1,  help='compress layer of model')
    parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')

    args = parser.parse_args()
    args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

    return args

def get_parameter():
    args  = build_parser()
    
    print("-*-" * 10 + "\n\t\tArguments\n" + "-*-" * 10)
    for key, value in vars(args).items():
        print("%s: %s" % (key, value))

    return args