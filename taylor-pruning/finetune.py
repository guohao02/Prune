import argparse
import copy
import os
# import cv2
import sys
import time
from heapq import nsmallest
from operator import itemgetter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torchvision import models

import dataset
from prune import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class ModifiedVGG16Model(torch.nn.Module):#vgg16模型
    def __init__(self):
        super(ModifiedVGG16Model, self).__init__()

        model = models.vgg16(pretrained=True)
        self.features = model.features

        for param in self.features.parameters():
            param.requires_grad = False
        #采用VGG16，丢弃完全连接的层，并添加三个新的完全连接的层
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)#扁平化
        x = self.classifier(x)
        return x


class FilterPrunner:#卷积核剪枝
    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        # self.activations = []
        # self.gradients = []
        # self.grad_index = 0
        # self.activation_to_layer = {}
        self.filter_ranks = {}#空字典

    def forward(self, x):#向前传播
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}

        activation_index = 0#激活索引
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)#module输出：例：Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)#钩子，register_hook的作用：即对x求导时，对x的导数进行操作，并且register_hook的参数只能以函数的形式传过去，即插入操作。
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1#激活索引

        return self.model.classifier(x.view(x.size(0), -1))

    # def compute_rank(self, activation_index):
    #     # Returns a partial function
    #     # as the callback function
    #     def hook(grad):
    #         activation = self.activations[activation_index]#输出对应的层
    #         # print((activation * grad).shape)
    #         values = \
    #             torch.sum((activation * grad), dim=0, keepdim=True).\
    #              sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[0, :, 0, 0].data#activation乘以梯度的除第二个维度，其他维度的累加
    #         #    sum(dim=2).sum(dim=3)[0, :, 0, 0].data

    #         # Normalize the rank by the filter dimensions
    #         values = \
    #             values / (activation.size(0) * activation.size(2)
    #                       * activation.size(3))#平均，values中的元素个数是通道数个

    #         if activation_index not in self.filter_ranks:
    #             self.filter_ranks[activation_index] = \
    #                 torch.FloatTensor(activation.size(1)).zero_().cuda()#生成activation.size(1)列的向量

    #         self.filter_ranks[activation_index] += values#更新values
    #         self.grad_index += 1#梯度索引
    #     return hook
    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1#从后往前backforWord
        activation = self.activations[activation_index]

        taylor = activation * grad
        # Get the average value for every filter, 
        # accross all the other dimensions
        taylor = taylor.mean(dim=(0, 2, 3)).data


        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = \
                torch.FloatTensor(activation.size(1)).zero_()

            if args.use_cuda:
                self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

        self.filter_ranks[activation_index] += taylor
        # print("filter_ranks",taylor)
        self.grad_index += 1
        # print("self.grad_index",self.grad_index)

    def lowest_ranking_filters(self, num):#返回value前num个最小的通道,#被get_prunning_plan调用，计算最小的512个filter
        data = []
        for i in sorted(self.filter_ranks.keys()):#按键值排序
            for j in range(self.filter_ranks[i].size(0)):#通道数
                data.append(
                    (self.activation_to_layer[i], j, self.filter_ranks[i][j]))#栈中压入，对应索引的layer，对应的通道，对应键值和通道的value

        return nsmallest(num, data, itemgetter(2))#nlargest和nsmallest在某个集合中找出最大或最小的N个元素，itemgetter(2)获取对象的第3个域的值(value值)

    def normalize_ranks_per_layer(self):#标准化
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i]).cpu()
            v = v / np.sqrt(torch.sum(v * v))
            self.filter_ranks[i] = v.cpu()

    def get_prunning_plan(self, num_filters_to_prune):#被PrunningFineTuner_VGG16类调用
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)#要剪去的通道

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:#l为层索引，f为通道索引
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(
                filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i #https://blog.csdn.net/weixin_42071277/article/details/90023105

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune#剪枝通道的索引


class PrunningFineTuner_VGG16:
    def __init__(self, train_path, test_path, model):#初始化
        self.train_data_loader = dataset.loader(train_path)
        self.test_data_loader = dataset.test_loader(test_path)

        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()
        self.prunner = FilterPrunner(self.model)
        self.model.train()

    def test(self):#测试
        self.model.eval()
        correct = 0
        total = 0

        for i, (batch, label) in enumerate(self.test_data_loader):
            batch = batch.cuda()
            output = model(Variable(batch))
            # print(output.data)
            pred = output.data.max(1)[1]# troch.max(1)[1]，只返回第二个维度最大值的每个索引;torch.max()[0]， 只返回最大值的每个数
            correct += pred.cpu().eq(label).sum()#正确数
            total += label.size(0)#总数

        print("Accuracy :" + str(float(correct) / total))

        self.model.train()

    def train(self, optimizer=None, epoches=10):#训练
        if optimizer is None:
            optimizer = \
                optim.SGD(model.classifier.parameters(),
                          lr=0.0001, momentum=0.9)#优化器

        for i in range(epoches):
            print("Epoch: ", i)
            self.train_epoch(optimizer)
            self.test()
        print("Finished fine tuning.")

    def train_batch(self, optimizer, batch, label, rank_filters):
        self.model.zero_grad()#梯度值初始化
        input = Variable(batch)#将tensor转为Variable

        if rank_filters:#train阶段为false，prun阶段为true
            output = self.prunner.forward(input)#调用FilterPrunner的forward，会计算用于rank的值
            self.criterion(output, Variable(label)).backward()#criterion调用CrossEntropyLoss()计算loss，再loss.backward()进行误差回传
        else:
            #self.model是VGG16，计算的是VGG16的output与lable的CrossEntropyLoss(交叉熵)
            self.criterion(self.model(input), Variable(label)).backward()
            optimizer.step()

    def train_epoch(self, optimizer=None, rank_filters=False):
        for batch, label in self.train_data_loader:
            self.train_batch(optimizer, batch.cuda(),
                             label.cuda(), rank_filters)

    def get_candidates_to_prune(self, num_filters_to_prune):
        self.prunner.reset()

        self.train_epoch(rank_filters=True)

        self.prunner.normalize_ranks_per_layer()

        return self.prunner.get_prunning_plan(num_filters_to_prune)#获取剪枝计划

    def total_num_filters(self):#通道数总数
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def prune(self):
        # Get the accuracy before prunning
        self.test()#测试

        self.model.train()

        # Make sure all the layers are trainable
        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()#通道数
        num_filters_to_prune_per_iteration = 512#要剪枝数
        iterations = int(float(number_of_filters) /
                         num_filters_to_prune_per_iteration)

        iterations = int(iterations * 2.0 / 3)#迭代次数

        print("Number of prunning iterations to reduce 67% filters", iterations)

        for _ in range(iterations):
            print("Ranking filters.. ")
            prune_targets = self.get_candidates_to_prune(
                num_filters_to_prune_per_iteration)#获取剪枝计划
            layers_prunned = {}
            for layer_index, filter_index in prune_targets:
                if layer_index not in layers_prunned:
                    layers_prunned[layer_index] = 0
                layers_prunned[layer_index] = layers_prunned[layer_index] + 1

            print("Layers that will be prunned", layers_prunned)
            print("Prunning filters.. ")
            model = self.model.cpu()
            for layer_index, filter_index in prune_targets:
                model = prune_vgg16_conv_layer(
                    model, layer_index, filter_index)#Vgg16卷积层剪枝

            self.model = model.cuda()

            message = str(100 * float(self.total_num_filters()) /number_of_filters) + "%"#剪枝率
            print("Filters prunned", str(message))
            self.test()
            print("Fine tuning to recover from prunning iteration.")
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=0.001, momentum=0.9)
            self.train(optimizer, epoches=10)

        print("Finished. Going to fine tune the model a bit more")
        self.train(optimizer, epoches=15)
        torch.save(model, "model_prunned")


def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--train", dest="train", action="store_true")
    # parser.add_argument("--prune", dest="prune", action="store_true")
    parser.add_argument("--train", action='store_true',
                    help='flag for training network', default= False)
    parser.add_argument("--prune", action='store_true',
                    help='flag for pruning network', default= True)
    parser.add_argument("--train_path", type=str, default="train")
    parser.add_argument("--test_path", type=str, default="test")
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    # parser.set_defaults(train=False)
    # parser.set_defaults(prune=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    if args.train:
        model = ModifiedVGG16Model().cuda()
    elif args.prune:
        model = torch.load("model").cuda()

    fine_tuner = PrunningFineTuner_VGG16(
        args.train_path, args.test_path, model)

    if args.train:
        fine_tuner.train(epoches=20)
        torch.save(model, "model")

    elif args.prune:
        fine_tuner.prune()
    # train_data_loader = dataset.loader(args.train_path)
    # model = ModifiedVGG16Model().cuda()
    # prunner= FilterPrunner(model)
    # for batch, label in train_data_loader:
    #     model.zero_grad()
    #     input = Variable(batch.cuda())
    #     x = prunner.forward(input)
    #     print(x)
