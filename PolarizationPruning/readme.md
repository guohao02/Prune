# 剪枝——结构化剪枝
## 下面要介绍的论文为：
[Neuron-level Structured Pruning using Polarization Regularizer](https://www.researchgate.net/profile/Tao-Zhuang-4/publication/344781579_Neuron-level_Structured_Pruning_using_Polarization_Regularizer/links/5f9bf48c299bf1b53e514c0c/Neuron-level-Structured-Pruning-using-Polarization-Regularizer.pdf)(2020)  
## 介绍
该篇文章是2017年network slimming的改进工作。在network slimming的工作中，利用L1正则化技术让BN层的scale系数趋近于0，然后裁剪“不重要”的channel。然而，这篇文章认为这种做法存在一定问题。L1正则化技术会让所有的scale系数都趋近于0，更理想的做法应该是只减小“不重要”channel的scale系数，保持其他系数的仍处于较大状态。为了实现这一想法，该篇文章提出了polarization正则化技术，使scale系数两极化。  
![image](https://user-images.githubusercontent.com/80331072/116501567-af7a4300-a8e3-11eb-80d6-0e95041e9d7c.png)

如图所示，可以看出L1正则化的scale系数分布和polarization正则化的scale系数分布。Polarization正则化技术能够更准确的确定裁剪阈值  
## 方法
由于BN在现代神经网络中被广泛采用，文章中使用BN中的比例因子作为每个神经元的比例因子。尺度因子正则化的网络训练目标函数为:
![image](https://user-images.githubusercontent.com/80331072/116501782-4c3ce080-a8e4-11eb-9c92-7f3d4f47858b.png)

其中L()为损失函数，R()通常为网络权值的L2正则化，Rs()为神经元尺度因子的稀疏正则化。在剪枝过程中，选择一个阈值，对低于阈值的神经元进行剪枝。在[Liu et al.， 2017]中，选择稀疏正则化器为L1，即Rs(γ) = ||γ||1。L1正则化的效果是将所有缩放参数推到0。因此，L1正则化缺乏修剪和保留神经元之间的区分。更合理的剪枝方法是只抑制不重要的神经元(缩放因子为0)，同时保留重要的神经元(缩放因子较大)。为了实现这一目标，文章中提出了一种新的基于比例因子的正则化器，即极化正则化器。

### 极化正则化器
为了防止比例因子收敛到一个值,将极化调节器定义为
![image](https://user-images.githubusercontent.com/80331072/116520774-bd40c000-a905-11eb-9c8b-be37f13acc8a.png)

图中所示的等式在L1项中增加新项![image](https://user-images.githubusercontent.com/80331072/116519119-a9945a00-a903-11eb-8e8f-3b8808ba70fb.png)
其作用是将γ尽可能与平均值分离，实现γ两极化分布。
最优解：等式中最优解分布为nρ或者nρ+1个γ是a，剩下的是0
![image](https://user-images.githubusercontent.com/80331072/116521258-553ea980-a906-11eb-829a-1726a66485c3.png)

## 思路
通过极化正则化，将BN层的缩放因子推向两极化，由此可以更加准确的获得剪枝的阈值，只剪去那些不重要的通道。

## 核心代码
```

