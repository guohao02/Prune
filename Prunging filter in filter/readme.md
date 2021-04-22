# 剪枝——Prunging filter in filter
## 下面要介绍的论文为：
[Prunging filter in filter](https://arxiv.org/pdf/2009.14410.pdf)  

## 思路
引入FS层，FS是为了学习滤波器除了权重之外的另一个重要特性:形状，这是一个与滤波器stripes相关的矩阵。FS中的每个值对应于滤波器中的一个stripes。每层的FS首先用全一矩阵进行初始化。在训练过程中，我们将过滤器的权重与FS相乘。数学上的，损失表示为：
![image](https://user-images.githubusercontent.com/80331072/115669934-6e31e280-a37b-11eb-875f-67819248735f.png)

其中W为卷积核权值，I为FS层，x为输入特征图。其前向传播过程为：  
![image](https://user-images.githubusercontent.com/80331072/115670202-bc46e600-a37b-11eb-9c2d-a19f22d54eae.png)
可获得W和I的梯度为：  
![image](https://user-images.githubusercontent.com/80331072/115670303-da144b00-a37b-11eb-9225-be29c32eb117.png)
计算过程如图所示：  
![image](https://user-images.githubusercontent.com/80331072/115670503-1051ca80-a37c-11eb-8bb4-58335c2a7bb0.png)
