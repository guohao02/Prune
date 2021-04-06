from __future__ import division

import numpy as np
import models
import os, sys, shutil, time, random
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from parameter_softprune import get_parameter
from soft_pruning import Mask



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def seed(args):
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.benchmark = True
    
    # Init logger
def logger(args):
    if not os.path.isdir(args.save_path):#os.path.isdir用于判断对象是否为一个目录
        os.makedirs(args.save_path)#创建目录
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}#返回args中每个元素
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])#compose把多个图像处理操作集合在一起，参考链接：https://blog.csdn.net/u013925378/article/details/103363232
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                                num_workers=args.workers, pin_memory=True)

    print_log("=> creating model '{}'".format(args.arch), log)
    # Init model, criterion, and optimizer
    net = models.__dict__[args.arch](num_classes)#初始化模型
    print_log("=> network :\n {}".format(net), log)
    return net ,train_loader, test_loader,log,state


    # define loss function (criterion) and optimizer
def criterion_optimizer(args,net,state):
    criterion = torch.nn.CrossEntropyLoss()#交叉熵主要是用来判定实际的输出与期望的输出的接近程度,交叉熵的值越小，两个概率分布就越接近

    optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                                    weight_decay=state['decay'], nesterov=True)#优化函数
    if args.use_cuda:
        net = net.cuda()
        criterion =criterion.cuda()
    return net,criterion,optimizer

def resume_f_checkpoint(args, log):
    recorder = RecorderMeter(args.epochs)
    # optionally resume from a checkpoint
    if args.resume:#加载训练模型
        if os.path.isfile(args.resume):
            print_log("=> loading checkpoint '{}'".format(args.resume), log)
            checkpoint = torch.load(args.resume)#加载训练模型
            recorder = checkpoint['recorder']
            args.start_epoch = checkpoint['epoch']
            if args.use_state_dict:
                net.load_state_dict(checkpoint['state_dict'])
            else:
                net = checkpoint['state_dict']
                
            optimizer.load_state_dict(checkpoint['optimizer'])
            print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
        else:
            print_log("=> no checkpoint found at '{}'".format(args.resume), log)
    else:
        print_log("=> do not use any checkpoint for {} model".format(args.arch), log)
    return recorder

    if args.evaluate:#评估模型
        time1 = time.time()
        validate(test_loader, net, criterion, log)
        time2 = time.time()
        print ('function took %0.3f ms' % ((time2-time1)*1000.0))
    return

    # Main loop
def main_loop(m,args,optimizer,train_loader,test_loader,net,criterion,recorder,comp_rate,log):
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(args.start_epoch, args.epochs):
        current_learning_rate = adjust_learning_rate(args, optimizer, epoch, args.gammas, args.schedule)

        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

        print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
                                + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

        # train for one epoch
        train_acc, train_los = train(args,train_loader, net, criterion, optimizer, epoch, log)

        # evaluate on validation set
        val_acc_1,   val_los_1   = validate(args,test_loader, net, criterion, log)
        if (epoch % args.epoch_prune ==0 or epoch == args.epochs-1):

            m.model = net
            m.if_zero()
            m.init_mask(comp_rate)
            m.do_mask()
            m.if_zero()
            net = m.model 
            if args.use_cuda:
                net = net.cuda()  
            
        val_acc_2,   val_los_2   = validate(args,test_loader, net, criterion, log)
    
        
        is_best = recorder.update(epoch, train_los, train_acc, val_los_2, val_acc_2)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': net,
            'recorder': recorder,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.save_path, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        #recorder.plot_curve( os.path.join(args.save_path, 'curve.png') )
    log.close()


def train(args,train_loader, model, criterion, optimizer, epoch, log):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):#将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)#包装Tensor，将Tensor转换为Variable之后，可以装载梯度信息
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                        'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                        epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
    print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
    return top1.avg, losses.avg

def validate(args,val_loader, model, criterion, log):#test
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))#[0]
        top1.update(prec1, input.size(0))#[0]
        top5.update(prec5, input.size(0))#[0]

    print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

    return top1.avg, losses.avg

def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def save_checkpoint(state, is_best, save_path, filename):#保存最好的模型
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)

def adjust_learning_rate(args, optimizer, epoch, gammas, schedule):#自适应学习率
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):#精度
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)#.reshape.view
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def main():
    args = get_parameter()
    
    seed(args)
    net ,train_loader, test_loader, log,state = logger(args)
    net ,criterion ,optimizer = criterion_optimizer(args, net = net, state = state)
    recorder = resume_f_checkpoint(args,log= log)

    m=Mask(net,args)#加载resnet,实例化
    m.init_length()
    comp_rate =  args.rate
    print("-"*10+"one epoch begin"+"-"*10)
    print("the compression rate now is %f" % comp_rate)

    val_acc_1, val_los_1 = validate(args, val_loader= test_loader ,model = net, criterion = criterion, log =log)
    print(" accu before is: %.3f %%" % val_acc_1)
    
    m.model = net 
    m.init_mask(comp_rate)
#    m.if_zero()
    m.do_mask()
    net = m.model
#    m.if_zero()
    if args.use_cuda:
        net = net.cuda()    
    val_acc_2,   val_los_2   = validate(args, test_loader, net, criterion, log)
    print(" accu after is: %s %%" % val_acc_2)
    main_loop(m, args,optimizer,train_loader,test_loader,net,criterion,recorder,comp_rate, log)

if __name__ == '__main__':  
    main()

