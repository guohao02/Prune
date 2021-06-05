# 剪枝——Pruning filter in filter
## 下面要介绍的论文为：
[Pruning filter in filter](https://arxiv.org/pdf/2009.14410.pdf)  (NeurIPS 2020)

## 方法
### 引入FS层  
FS是为了学习滤波器除了权重之外的另一个重要特性:形状，这是一个与滤波器stripes相关的矩阵。FS中的每个值对应于滤波器中的一个stripes。每层的FS首先用全一矩阵进行初始化。在训练过程中，我们将过滤器的权重与FS相乘。数学上的，损失表示为：
![image](https://user-images.githubusercontent.com/80331072/115669934-6e31e280-a37b-11eb-875f-67819248735f.png)

其中W为卷积核权值，I为FS层，x为输入特征图。其前向传播过程为：  
![image](https://user-images.githubusercontent.com/80331072/115670202-bc46e600-a37b-11eb-9c2d-a19f22d54eae.png)

可获得W和I的梯度为：  
![image](https://user-images.githubusercontent.com/80331072/115670303-da144b00-a37b-11eb-9225-be29c32eb117.png)

### 计算过程如图所示：  
![image](https://user-images.githubusercontent.com/80331072/115674580-6294ea80-a380-11eb-846d-03521db9517b.png)
**思路：**
从图中可以看到，FS层是学习每个滤波器最优的形状。为了实现高效的剪枝，我们设置了阈值δ，在训练过程中，FS值小于δ的stripe不会被更新，可以在训练后进行剪枝。值得注意的是，在对修剪后的网络进行推理时，由于滤波器被破坏，不能直接使用滤波器作为一个整体对输入特征图进行卷积。相反，我们需要单独使用每个stripe来执行卷积，并对每个stripe生成的特征地图求和。如果将该滤波器中的所有stripe从网络中去除，我们也不需要记录该滤波器的索引，SWP退化为传统的滤波器式剪枝。数学上的SWP卷积过程为：  
![image](https://user-images.githubusercontent.com/80331072/115674535-5446ce80-a380-11eb-8445-bf843a1a66d2.png)

### 算法创新点
(1) 提出滤波器除了参数属性外，还存在形状属性，并且形状属性具有重要意义。

(2) 提出滤波器骨架的模块来学习滤波器的形状，并可以指导模型剪枝。

(3) 通过变换普通卷积为Stripe-Wise Convolution，结构化的实现逐条剪枝后的模型。

## 核心代码
```
# flops.py
    def filter_strip_hook(self, input, output):重构了stripe-wise-convolutional的flops计算方法
# vgg.py（resnet56也类似）
    def __init__(self, num_classes=10, cfg=None):
        ...
        self.classifier = Linear(512, num_classes)# 将线性层换成一键裁剪的线性层
    def _make_layers(self, cfg):
        ...
        layers += [FilterStripe(in_channels, x),# 将卷积层换成一键转换逐条卷积的卷积层
                   BatchNorm(x),---将BN层换成一键裁剪的BN层
                   nn.ReLU(inplace=True)]
        ...
    def update_skeleton(self, sr, threshold):# 训练过程中更新模型的形状（置0方式）

    def prune(self, threshold):# 训练完成后真正实现逐条裁剪
# stripe.py
    class FilterStripe(nn.Conv2d):#卷积+FS层
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
            super(FilterStripe, self).__init__(in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=1, bias=False)
            self.BrokenTarget = None# 状态标志，训练时为None，训练完裁剪后为裁剪的情况
            self.FilterSkeleton = Parameter(torch.ones(self.out_channels, self.kernel_size[0], self.kernel_size[1]), requires_grad=True)# 滤波器骨架
    
        def forward(self, x):
            if self.BrokenTarget is not None:# 逐条裁剪后结构化实现方式
                out = torch.zeros(x.shape[0], self.FilterSkeleton.shape[0], int(np.ceil(x.shape[2] / self.stride[0])), int(np.ceil(x.shape[3] / self.stride[1])))
                if x.is_cuda:
                    out = out.cuda()
                x = F.conv2d(x, self.weight)# 首先将所有条和feature map相乘。
                l, h = 0, 0
                for i in range(self.BrokenTarget.shape[0]):
                    for j in range(self.BrokenTarget.shape[1]):
                        h += self.FilterSkeleton[:, i, j].sum().item()
                        out[:, self.FilterSkeleton[:, i, j]] += self.shift(x[:, l:h], i, j)[:, :, ::self.stride[0], ::self.stride[1]]# 然后按照相应位置shift后相加。
                        l += self.FilterSkeleton[:, i, j].sum().item()
                return out
            else:
                return F.conv2d(x, self.weight * self.FilterSkeleton.unsqueeze(1), stride=self.stride, padding=self.padding, groups=self.groups)# 正常训练时将参数与骨架乘在一起，通过置0屏蔽无效stripes。
    
        def prune_in(self, in_mask=None):# 裁剪输入通道
        
        def prune_out(self, threshold):# 裁剪输出通道
    
        def _break(self, threshold):# 将正常卷积转换为逐条剪枝的参数移植
            self.weight = Parameter(self.weight * self.FilterSkeleton.unsqueeze(1))
            self.FilterSkeleton = Parameter((self.FilterSkeleton.abs() > threshold), requires_grad=False)
            if self.FilterSkeleton.sum() == 0:
                self.FilterSkeleton.data[0][0][0] = True
            self.out_channels = self.FilterSkeleton.sum().item()
            self.BrokenTarget = self.FilterSkeleton.sum(dim=0)
            self.kernel_size = (1, 1)
            self.weight = Parameter(self.weight.permute(2, 3, 0, 1).reshape(-1, self.in_channels, 1, 1)[self.FilterSkeleton.permute(1, 2, 0).reshape(-1)])
    
        def update_skeleton(self, sr, threshold):# 训练时更新滤波器骨架
            self.FilterSkeleton.grad.data.add_(sr * torch.sign(self.FilterSkeleton.data))# l1 norm惩罚
            mask = self.FilterSkeleton.data.abs() > threshold # 屏蔽掉小于阈值的stripes
            self.FilterSkeleton.data.mul_(mask)
            self.FilterSkeleton.grad.data.mul_(mask)
            out_mask = mask.sum(dim=(1, 2)) != 0
            return out_mask
    
        def shift(self, x, i, j):# 3*3卷积上每个点扫描到的特征图是不同的，不同位置需要对应平移才能相加
            return F.pad(x, (self.BrokenTarget.shape[0] // 2 - j, j - self.BrokenTarget.shape[0] // 2, self.BrokenTarget.shape[0] // 2 - i, i - self.BrokenTarget.shape[1] // 2), 'constant', 0)
```
## 代码运行
```
mkdir -p checkpoint/VGG/sr0.00001_threshold_0.01
python main.py --arch VGG --data_path ../data --sr 0.00001 --threshold 0.01 --save checkpoint/VGG/sr0.00001_threshold_0.01
```
## 代码参考
[fxmeng/Pruning-Filter-in-Filter](https://github.com/fxmeng/Pruning-Filter-in-Filter)

