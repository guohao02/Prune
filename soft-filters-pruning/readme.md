# 剪枝--soft-filters-pruning
## 下面要介绍的论文为：  
[Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/pdf/1808.06866.pdf)  

## SFP算法
### 算法的流程
**流程图**  
![image](https://user-images.githubusercontent.com/80331072/113682117-5c9ed880-96f5-11eb-9657-8d9549003c10.png)  
在上图中对第k次epoch进行剪枝，检测根据是L-norm进行度量，代码中采用的的是L2，剪除权重较小的filters（置0）。在下个epoch中进行迭代。 

**L2-norm公式：**  
![image](https://user-images.githubusercontent.com/80331072/113713905-70a90100-971a-11eb-9bff-0cc8344c2b31.png)

### 算法的具体步骤  
![image](https://user-images.githubusercontent.com/80331072/113683581-f31fc980-96f6-11eb-9103-fb741e8c7bb6.png) 
**如上图所示：**   
(1)初始化模型权重  
(2)每个epoch更新模型权重  
&nbsp; <1>将每个卷积核的权重的绝对值相加，从小到大排序  
&nbsp; <2>剪除前NP个卷积核(置0)  
&nbsp; <3>返回到(2)，继续训练  
(3)获取最优的模型参数，并返回模型  

### 算法优点：
相比较于hard filter pruning，soft filter pruning 允许模型从随机初始化开始（从预训练模型开始能获得更好的效果），并在每个epoch训练开始之前，将具有较小L2-norm的filters置零，然后更新所有filters（包括未剪枝和已剪枝filters），最终模型收敛以后再把一些不重要的filters（zero-filters）裁剪掉，从而获得模型容量较高、推理精度较高的正则化、剪枝结果。

## 核心代码
```
    def get_codebook(self, weight_torch,compress_rate,length):#获取掩码
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()
    
        weight_abs = np.abs(weight_np)#获取权值的绝对值
        weight_sort = np.sort(weight_abs)#排序
        
        threshold = weight_sort[int (length * (1-compress_rate) )]#阈值
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

            kernel_length = weight_torch.size()[1] *weight_torch.size()[2] *weight_torch.size()[3]#卷积核的大小
            for x in range(0,len(filter_index)):
                codebook [filter_index[x] *kernel_length : (filter_index[x]+1) *kernel_length] = 0#置零

            print("filter codebook done")
        else:
            pass
        return codebook
```

## 代码运行
soft-pruning.py是剪枝文件，里面包含Mask类，可以在main.py中调用。main.py中包含数据集和训练集下载、train模块、test模块、最优模型数据保存模块等。代码中采用的模型是resnet20(models文件夹里面也有其他类型的网络模型供选择),采用的数据集为cifar10，首先初始化模型参数，进行剪枝和评估；再进入主循环中，先训练模型，再进行模型剪枝，直到完成所有的epoch。

```
python main.py
```
## 运行结果
### 对应剪枝层的权重
![image](https://user-images.githubusercontent.com/80331072/113711644-c9c36580-9717-11eb-8034-eeddfed2c51e.png)
### 准确率
![image](https://user-images.githubusercontent.com/80331072/113712064-41919000-9718-11eb-940c-75a5dda655e4.png)

