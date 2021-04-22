# 剪枝——Prunging filter in filter
## 下面要介绍的论文为：
[Prunging filter in filter](https://arxiv.org/pdf/2009.14410.pdf)  (NeurIPS 2020)

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
从图中可以看到，FS层是学习每个滤波器最优的形状。为了实现高效的剪枝，我们设置了阈值δ，在训练过程中，FS值小于δ的stripe不会被更新，可以在训练后进行剪枝。值得注意的是，在对修剪后的网络进行推理时，由于滤波器被破坏，不能直接使用滤波器作为一个整体对输入特征图进行卷积。相反，我们需要单独使用每个stripestripe来执行卷积，并对每个stripe生成的特征地图求和。如果将该滤波器中的所有stripe从网络中去除，我们也不需要记录该滤波器的索引，SWP退化为传统的滤波器式剪枝。数学上的SWP卷积过程为：  
![image](https://user-images.githubusercontent.com/80331072/115674535-5446ce80-a380-11eb-8445-bf843a1a66d2.png)

### 算法创新点
(1) 提出滤波器除了参数属性外，还存在形状属性，并且形状属性具有重要意义。

(2) 提出滤波器骨架的模块来学习滤波器的形状，并可以指导模型剪枝。

(3) 通过变换普通卷积为Stripe-Wise Convolution，结构化的实现逐条剪枝后的模型。

## 核心代码

