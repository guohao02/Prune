# taylor - pruning
## 下面介绍的论文为：  
[Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440)   
这篇论文来自NVIDIA，论文中提出了一种用于修剪神经网络中的卷积核的新公式，以实现有效的推理。  
论文中提出了一个基于泰勒展开的新准则，用它去近似由于修剪网络参数引起的损失函数的变化。  
## 主要思想
### 流程图如下所示：  
![image](https://user-images.githubusercontent.com/80331072/114808562-1e7f7400-9ddb-11eb-96ca-7ef0f5946b4f.png)  
修剪方法包括以下步骤：  
1）微调网络直到目标任务收敛。  
2）交替迭代修剪和进一步微调。  
3）在达到准确度和修剪目标之间的目标折衷之后停止修剪。  
### 修剪标准：使用泰勒展开进行修剪标准的衡量  
论文中使用下图这个函数去衡量修剪的W是否达到最优，即寻找一个修剪后效果最接近预训练好原始权重的W，使得损失函数变化最小。论文中认为找到一个好的W，同时保持尽可能接近原始成本值。  
![image](https://user-images.githubusercontent.com/80331072/114808889-bd0bd500-9ddb-11eb-8689-b9f575fd5602.png)
#### Taylor 展开：
修剪前后的损失变化用下面这个公式表示，hi=0代表的是修剪之后的损失。  
![image](https://user-images.githubusercontent.com/80331072/114809170-458a7580-9ddc-11eb-9062-79f99f520a35.png)  
使用1阶泰勒展开去逼近∆C(hi)，去掉高阶项，最后得到：  
![image](https://user-images.githubusercontent.com/80331072/114809276-710d6000-9ddc-11eb-98ff-be7fac126726.png)  

### 思路
获取每个通道的一阶泰勒项的绝对值，再对其进行排序。如果要剪去nums个，则每次取前nums小的一阶泰勒项对应的通道。再进行微调，直到满足要剪枝的比例。 

## 代码分析
```
def compute_rank(self, grad):#通过钩子获得，每个通道的一阶泰勒项
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
```
```
def lowest_ranking_filters(self, num):#返回value前num个最小的通道,#被get_prunning_plan调用，计算最小的512个filter
        data = []
        for i in sorted(self.filter_ranks.keys()):#按键值排序
            for j in range(self.filter_ranks[i].size(0)):#通道数
                data.append(
                    (self.activation_to_layer[i], j, self.filter_ranks[i][j]))#栈中压入，对应索引的layer，对应的通道，对应键值和通道的value

        return nsmallest(num, data, itemgetter(2))#nlargest和nsmallest在某个集合中找出最大或最小的N个元素，itemgetter(2)获取对象的第3个域的值(value值)
```
```
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
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i #减i是为了更新每次剪枝后的通道索引

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune#剪枝通道的索引
```
prune.py是获取剪枝计划后的剪枝过程

## 代码运行
train:
```
python finetune.py --train
```
prune:
```
python finetune.py --prune
```




