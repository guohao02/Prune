# 剪枝——结构化剪枝
## 下面要介绍的论文为：
[SNIP: SINGLE-SHOT NETWORK PRUNING BASED ON CONNECTION SENSITIVITY](https://arxiv.org/pdf/1810.02340.pdf)(ICLR 2019)
# 介绍
这篇文章提出了基于权重连接重要性（saliency）的剪枝方法，能够在深度模型训练之前（初始化阶段），通过mini-batch的多次采样，决定不同权重连接的重要性，进而根据剪枝目标生成剪枝模板（prunning mask）、应用于稀疏剪枝，从而节省了相对耗时的剪枝-微调迭代周期。
## 方法
文章提出了一种新的saliency 衡量准则，即损失函数关于权重连接的梯度。首先，剪枝优化问题重新定义如下：  
![image](https://user-images.githubusercontent.com/80331072/119095014-53b75b80-ba44-11eb-9289-d2a6613f7deb.png)  
其中矩阵C表示深度网络的连接模板，数值1表示连接，数值0表示断接。C(j)从1变为0所引起的loss变化，可以反映权重连接的重要性，并进一步等价于loss关于c的梯度  
![image](https://user-images.githubusercontent.com/80331072/119096108-bf4df880-ba45-11eb-94af-640a7241de0d.png)  
在剪枝阶段，为了跨层比较S(j)，梯度g(j)的大小需要做标准化处理，然后根据剪枝目标可决定剪枝模板：  
![image](https://user-images.githubusercontent.com/80331072/119096394-210e6280-ba46-11eb-9add-e09676f3f3ea.png)  




