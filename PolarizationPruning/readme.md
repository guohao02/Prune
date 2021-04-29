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
BN层稀疏化函数：
```
def bn_sparsity(model, loss_type, sparsity, t, alpha,
                flops_weighted: bool, weight_min=None, weight_max=None):#BN层稀疏化，缩放因子被推向两极
    """

    :type model: torch.nn.Module
    :type alpha: float
    :type t: float
    :type sparsity: float
    :type loss_type: LossType
    """
    bn_modules = model.get_sparse_layers()

    if loss_type == LossType.POLARIZATION or loss_type == LossType.L2_POLARIZATION:
        # compute global mean of all sparse vectors
        n_ = sum(map(lambda m: m.weight.data.shape[0], bn_modules))#map() 会根据提供的函数对指定序列做映射
        sparse_weights_mean = torch.sum(torch.stack(list(map(lambda m: torch.sum(m.weight), bn_modules)))) / n_#获取Bn层权值的均值

        sparsity_loss = 0.
        if flops_weighted:
            for sub_module in model.modules():
                if isinstance(sub_module, model.building_block):
                    flops_weight = sub_module.get_conv_flops_weight(update=True, scaling=True)
                    sub_module_sparse_layers = sub_module.get_sparse_modules()

                    for sparse_m, flops_w in zip(sub_module_sparse_layers, flops_weight):
                        # linear rescale the weight from [0, 1] to [lambda_min, lambda_max]，将权重从[0，1]线性重缩放到[lambda_min，lambda_max]
                        flops_w = weight_min + (weight_max - weight_min) * flops_w

                        sparsity_term = t * torch.sum(torch.abs(sparse_m.weight.view(-1))) - torch.sum(
                            torch.abs(sparse_m.weight.view(-1) - alpha * sparse_weights_mean))
                        sparsity_loss += flops_w * sparsity * sparsity_term
            return sparsity_loss
        else:
            for m in bn_modules:
                if loss_type == LossType.POLARIZATION:
                    sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(
                        torch.abs(m.weight - alpha * sparse_weights_mean))
                elif loss_type == LossType.L2_POLARIZATION:
                    sparsity_term = t * torch.sum(torch.abs(m.weight)) - torch.sum(
                        (m.weight - alpha * sparse_weights_mean) ** 2)
                else:
                    raise ValueError(f"Unexpected loss type: {loss_type}")
                sparsity_loss += sparsity * sparsity_term

            return sparsity_loss
    else:
        raise ValueError()
```
剪枝函数：
```
def prune_model(self, pruner: Callable[[np.ndarray], float], prune_mode: str) -> None:
def do_pruning(self, in_channel_mask: np.ndarray, pruner: Callable[[np.ndarray], float], prune_mode: str):
def prune_conv_layer()
```
获取阈值函数：
```
def search_threshold(weight: np.ndarray, alg: str):
    if alg not in ["fixed", "grad", "search"]:
        raise NotImplementedError()

    hist_y, hist_x = np.histogram(weight, bins=100, range=(0, 1))
    if alg == "search":
        raise ValueError(f"Deprecated pruning algorithm: {alg}")
    elif alg == "grad":
        hist_y_diff = np.diff(hist_y)
        for i in range(len(hist_y_diff) - 1):
            if hist_y_diff[i] <= 0 <= hist_y_diff[i + 1]:
                threshold = hist_x[i + 1]
                if threshold > 0.2:
                    print(f"WARNING: threshold might be too large: {threshold}")
                return threshold
    elif alg == "fixed":
        return hist_x[1]
```
