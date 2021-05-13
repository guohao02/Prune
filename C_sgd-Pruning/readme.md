# 剪枝——结构化剪枝
## 下面要介绍的论文为：
[Centripetal SGD for Pruning Very Deep Convolutional Networks withComplicated Structure](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.pdf)(CVPR2020)
## 介绍
本文提出了一种创新的优化方法——Centripetal SGD(C-SGD)。顾名思义，我们将多个卷积核移动到卷积核参数的超空间的中心。同时，根据模型的原始目标函数，尽可能地保持模型的性能。通过C-SGD方法训练一些卷积核使其变得相同，从而在CNNs中产生冗余模式。与基于重要性的卷积核剪枝方法相比，这样做不需要对卷积核的重要性有启发性的认识。与置零法相比，该方法不需要进行微调，且保留了更有代表性的网络结构。  
![image](https://user-images.githubusercontent.com/80331072/118096150-90ab9e80-b403-11eb-89f2-6617e946b085.png)  
图中所示为归零与向心约束。这幅图显示了一个CNN，在第1和第2个卷积层分别有4和6个滤波器，它接受2通道的输入。  
左:在conv1处的第3个滤波器被置零，因此第3个feature map接近于0，这意味着在conv2处的6个滤波器的第3个输入通道是无用的。在剪枝过程中，去掉conv1处的第3个滤波器以及conv2处6个滤波器的第3个输入通道。  
右:conv1的第3和第4个滤镜由于向心约束被迫变得很近，直到第3和第4个特征图变得完全相同。但在conv2的6个滤波器的第3和第4个输入通道仍然可以不受约束地增长，使编码的信息仍然得到充分利用。当剪枝时，去掉conv1处的第4个滤波器，将每个conv2处滤波器的第4个输入通道添加到第3个通道。  
## 方法
对于每个卷积层，首先将卷积核分成簇。簇的数量等于所需的卷积核的数量，因为只为每个簇保留一个卷积核。可以使用均匀聚类或者k-means生成簇。  
**K-means聚类** 目标是在参数超空间中生成较小群内间距的簇，自然地可以将它们压缩到一个单独的点而更小地影响模型。为此，我们简化了卷积核的核，将其作为特征向量用来K-means聚类。  
**平均聚类** 可以在不考虑卷积核的固有性质的情况下生成簇。假设x和y是原有的卷积核数目和需要的簇的数目，那么每个簇将最多有个[x/y]卷积核.  

