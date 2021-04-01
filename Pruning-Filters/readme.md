# Pruning-Filters
# 剪枝--Pruning-Filters
## 下面介绍第二篇论文：
Pruning Filters for Efficient ConvNets

#### 主要思想：
由于CNN通常在不同的 Filter 和特征信道之间具有显着的冗余，论文中通过修剪 Filter 来减少CNN的计算成本。与在整个网络中修剪权重相比， Filter 修剪是一种自然结构化的修剪方法，不会引入稀疏性，因此不需要使用稀疏库或任何专用硬件。通过减少矩阵乘法的次数，修剪 Filter 的数量与加速度直接相关，这很容易针对目标加速进行调整。

![image](https://user-images.githubusercontent.com/80331072/112116080-dbbfe700-8bf4-11eb-8045-bac5bbc938c7.png)

如上图所示，删除一个 Filter 就能减少一个输出特征图，同时特征图对应的接下来卷积操作同样可以删除掉。

### 修剪Filter步骤：

1)计算 Filter 中所有权值的绝对值之和

2)根据求和大小排列 Filter

3)删除数值较小的 Filter （权重数值越小，代表权重的重要性越弱）

4)对删除之后的 Filter 重新组合，生成新的Filter矩阵

### 多层同时修剪：

作者给出了2中修剪思路：

1)独立修剪：修剪时每一层是独立的

2)贪心修剪：修剪时考虑之前图层中删除的 Filter 。

#### 两种方法的区别：
独立修剪在计算（求权重绝对值之和）时不考虑上一层的修剪情况，所以计算时下图中的黄点仍然参与计算；贪心修剪计算时不计算已经修剪过的，即黄点不参与计算。
结果证明第二种方法的精度高一些。

![image](https://user-images.githubusercontent.com/80331072/112116392-36594300-8bf5-11eb-89cb-968c580db546.png)
![image](https://user-images.githubusercontent.com/80331072/112116453-453ff580-8bf5-11eb-9d1c-4d929aca1d47.png)

### 代码分析
文件包含五个py文件：main.py; vgg.py ;train.py ;parameter.py ;hardprune.py  
1.main.py调用train_network()函数，获得训练模型，保存在args.save_path路径中  
2.在args.load_path路径下加载训练模型，在给定的通道中剪枝相应数量的通道  
其核心代码为：  
```
def get_channel_index(kernel, num_elimination, residue=None):#获取要剪枝通道的索引
    # get cadidate channel index for pruning
    # 'residue' is needed for pruning by 'independent strategy'

    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)#kernel.view(kernel.size(0), -1)将卷积核尺寸自适应转换为[通道数,X]，再绝对值按列相加
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)

    _, args = torch.sort(sum_of_kernel)#卷积核绝对值相加之和排序，并获取移动状况，即索引

    return args[:num_elimination].tolist()#转换为列表，前num_elimination个最小的索引


def index_remove(tensor, dim, index, removed=False):#更新张量（更新卷积层、BN层、全连接层的权值和偏置）
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())#列表化
    new_size = tensor.size(dim) - len(index)#通道数减去索引的长度，获得新的通道数
    size_[dim] = new_size#更新的通道数放在第二个维度中

    select_index = list(set(range(tensor.size(dim))) - set(index))#set()创建一个无序不重复元素集，可进行关系测试，删除重复数据
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))#dim:0表示按行索引，1表示按列索引，torch.index_select保留被索引的行或者列

    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor
```
首先通过get_channel_index()函数获取要剪枝的通道，在每个层的更新函数（get_new_conv()、get_new_norm()和get_new_linear()）中调用index_remove()函数去更新对应的权值和偏置,进而更新要剪枝的层。  

### 代码运行
#### Training
```
python main.py --train-flag --save-path ./trained_models/vgg.pth --epoch 300 --lr 0.1 --lr-milestone 100 200
```
#### Pruning
```
python main.py --hard-prune-flag --load-path ./trained_models/vgg.pth --save-path ./prunned_models/vgg.pth --prune-layers conv1 conv8 conv9 conv10 conv11 conv12 conv13 --prune-channels 32 256 256 256 256 256 256
```
#### Retraining
```
python main.py --train-flag --load-path ./prunned_models/vgg.pth --save-path ./trained_prunned_models/vgg.pth --epoch 20 --lr 0.001
```
### 代码运行结果





