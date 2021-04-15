# taylor - pruning
## 下面介绍的论文为：  
[Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440)   
这篇论文来自NVIDIA，论文中提出了一种用于修剪神经网络中的卷积核的新公式，以实现有效的推理。  
论文中提出了一个基于泰勒展开的新准则，用它去近似由于修剪网络参数引起的损失函数的变化。  
## 主要思想
流程图如下所示：  
![image](https://user-images.githubusercontent.com/80331072/114808562-1e7f7400-9ddb-11eb-96ca-7ef0f5946b4f.png)  
修剪方法包括以下步骤：  
1）微调网络直到目标任务收敛。  
2）交替迭代修剪和进一步微调。  
3）在达到准确度和修剪目标之间的目标折衷之后停止修剪。  
修剪标准：使用泰勒展开进行修剪标准的衡量  
论文中使用下图这个函数去衡量修剪的W是否达到最优，即寻找一个修剪后效果最接近预训练好原始权重的W，使得损失函数变化最小。论文中认为找到一个好的W，同时保持尽可能接近原始成本值。  
![image](https://user-images.githubusercontent.com/80331072/114808889-bd0bd500-9ddb-11eb-8689-b9f575fd5602.png)
Taylor 展开：
修剪前后的损失变化用下面这个公式表示，hi=0代表的是修剪之后的损失。  
![image](https://user-images.githubusercontent.com/80331072/114809170-458a7580-9ddc-11eb-9062-79f99f520a35.png)  
使用1阶泰勒展开去逼近∆C(hi)，去掉高阶项，最后得到：  
![image](https://user-images.githubusercontent.com/80331072/114809276-710d6000-9ddc-11eb-98ff-be7fac126726.png)




