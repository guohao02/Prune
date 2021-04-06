import torch
import numpy as np

class Mask:
    def __init__(self,model,args):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.mat = {}
        self.mask_index = []
        self.args = args
        self.device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        
    
    def get_codebook(self, weight_torch,compress_rate,length):#剪枝
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()
    
        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)
        
        threshold = weight_sort[int (length * (1-compress_rate) )]
        #pruning：绝对值大于阈值置1，小于阈值置0
        weight_np [weight_np <= -threshold  ] = 1
        weight_np [weight_np >= threshold  ] = 1
        weight_np [weight_np !=1  ] = 0
        
        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch,compress_rate,length):
        codebook = np.ones(length)
        if len( weight_torch.size())==4:
            filter_pruned_num = int(weight_torch.size()[0]*(1-compress_rate))#剪枝数
            weight_vec = weight_torch.view(weight_torch.size()[0],-1)#将矩阵转换为weight_torch.size()[0]行，X列(自适应的)
            norm2 = torch.norm(weight_vec,2,1)#按行求2范数
            norm2_np = norm2.cpu().numpy()#转换成np
            filter_index = norm2_np.argsort()[:filter_pruned_num]#numpy.argsort()返回的是数组值从小到大的索引值
#            norm1_sort = np.sort(norm1_np)
#            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] *weight_torch.size()[2] *weight_torch.size()[3]#卷积核的大小
            for x in range(0,len(filter_index)):
                codebook [filter_index[x] *kernel_length : (filter_index[x]+1) *kernel_length] = 0#置零

            print("filter codebook done")
        else:
            pass
        return codebook
    
    def convert2tensor(self,x):
        x = torch.FloatTensor(x)#类型转换, 将list ,numpy转化为tensor
        return x
    
    def init_length(self):
        for index, item in enumerate(self.model.parameters()):#parameters()给出参数的迭代器
            self.model_size [index] = item.size()
        
        for index1 in self.model_size:
            for index2 in range(0,len(self.model_size[index1])):
                if index2 ==0:
                    self.model_length[index1] = self.model_size[index1][0]
                else:
                    self.model_length[index1] *= self.model_size[index1][index2]
                    
    def init_rate(self, layer_rate):
        for index, item in enumerate(self.model.parameters()):
            self.compress_rate [index] = 1
        for key in range(self.args.layer_begin, self.args.layer_end + 1, self.args.layer_inter):
            self.compress_rate[key]= layer_rate
        #different setting for  different architecture
        if self.args.arch == 'resnet20':
            last_index = 57
        elif self.args.arch == 'resnet32':
            last_index = 93
        elif self.args.arch == 'resnet56':
            last_index = 165
        elif self.args.arch == 'resnet110':
            last_index = 327
        self.mask_index =  [x for x in range (0,last_index,3)]
#        self.mask_index =  [x for x in range (0,330,3)]
        
    def init_mask(self,layer_rate):
        self.init_rate(layer_rate)
        for index, item in enumerate(self.model.parameters()):
            if(index in self.mask_index):
                self.mat[index] = self.get_filter_codebook(item.data, self.compress_rate[index],self.model_length[index] )
                self.mat[index] = self.convert2tensor(self.mat[index])
                if self.args.use_cuda:
                    self.mat[index] = self.mat[index].cuda()
        print("mask Ready")

    def do_mask(self):
        for index, item in enumerate(self.model.parameters()):
            if(index in self.mask_index):
                a = item.data.view(self.model_length[index])
                b = a * self.mat[index]
                item.data = b.view(self.model_size[index])
        print("mask Done")

    def if_zero(self):
        for index, item in enumerate(self.model.parameters()):
#            if(index in self.mask_index):
            if(index ==0):
                a = item.data.view(self.model_length[index])
                b = a.cpu().numpy()
                
                print("number of nonzero weight is %d, zero is %d" %( np.count_nonzero(b),len(b)- np.count_nonzero(b)))