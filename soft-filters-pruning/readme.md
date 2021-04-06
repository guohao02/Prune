# 剪枝--soft-filters-pruning
## 下面要介绍的论文为：  
[Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks](https://arxiv.org/pdf/1808.06866.pdf)  

## SFP算法
### 算法的流程
流程图  
![image](https://user-images.githubusercontent.com/80331072/113682117-5c9ed880-96f5-11eb-9657-8d9549003c10.png)  
在上图中对第k次epoch进行剪枝，检测根据是L-norm进行度量，代码中采用的的是L1，剪除权重较小的filters（置0）。在下个epoch中进行迭代。  
### 算法的具体步骤  
![image](https://user-images.githubusercontent.com/80331072/113683581-f31fc980-96f6-11eb-9103-fb741e8c7bb6.png)  
(1)初始化模型权重  
(2)每个epoch更新模型权重
(3)将每个卷积核的权重的绝对值相加，从小到大排序，剪除前N_{i+1}P_{i}

