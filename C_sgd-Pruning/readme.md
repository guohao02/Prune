# 剪枝——结构化剪枝
## 下面要介绍的论文为：
[Centripetal SGD for Pruning Very Deep Convolutional Networks withComplicated Structure](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.pdf)(CVPR2020)
## 介绍
&emsp;&emsp;本文提出了一种创新的优化方法——Centripetal SGD(C-SGD)。顾名思义，我们将多个卷积核移动到卷积核参数的超空间的中心。同时，根据模型的原始目标函数，尽可能地保持模型的性能。通过C-SGD方法训练一些卷积核使其变得相同，从而在CNNs中产生冗余模式。与基于重要性的卷积核剪枝方法相比，这样做不需要对卷积核的重要性有启发性的认识。与置零法相比，该方法不需要进行微调，且保留了更有代表性的网络结构。 
    
![image](https://user-images.githubusercontent.com/80331072/118096150-90ab9e80-b403-11eb-89f2-6617e946b085.png)  
&emsp;&emsp;图中所示为归零与向心约束。这幅图显示了一个CNN，在第1和第2个卷积层分别有4和6个滤波器，它接受2通道的输入。  
左:在conv1处的第3个滤波器被置零，因此第3个feature map接近于0，这意味着在conv2处的6个滤波器的第3个输入通道是无用的。在剪枝过程中，去掉conv1处的第3个滤波器以及conv2处6个滤波器的第3个输入通道。  
右:conv1的第3和第4个滤镜由于向心约束被迫变得很近，直到第3和第4个特征图变得完全相同。但在conv2的6个滤波器的第3和第4个输入通道仍然可以不受约束地增长，使编码的信息仍然得到充分利用。当剪枝时，去掉conv1处的第4个滤波器，将每个conv2处滤波器的第4个输入通道添加到第3个通道。  
## 方法
### 聚类
&emsp;&emsp;对于每个卷积层，首先将卷积核分成簇。簇的数量等于所需的卷积核的数量，因为只为每个簇保留一个卷积核。可以使用均匀聚类或者k-means生成簇。  
**K-means聚类** 目标是在参数超空间中生成较小群内间距的簇，自然地可以将它们压缩到一个单独的点而更小地影响模型。为此，我们简化了卷积核的核，将其作为特征向量用来K-means聚类。  
**平均聚类** 可以在不考虑卷积核的固有性质的情况下生成簇。假设x和y是原有的卷积核数目和需要的簇的数目，那么每个簇将最多有个[x/y]卷积核.  
### C-SGD
**更新规则**  

![image](https://user-images.githubusercontent.com/80331072/118098355-5ee80700-b406-11eb-9189-f851ef3d9c68.png)

&emsp;&emsp;其中L是原始目标函数，τ是学习率，η是模型的原始权重衰减系数，∈是引入的唯一的超参数，叫做向心力  
&emsp;&emsp;公式原理：对于在同一簇的卷积核，由目标函数得到的增量（第一项）被平均，第二项是普通的权重衰减，初始值的差异逐渐消除（最后一项），所以卷积核逐渐向它们在超空间的中心靠近。  
&emsp;&emsp;假设L是层的索引集，我们使用平方核偏差χ来度量簇内相似度，也就是簇内的卷积核有多相似
![image](https://user-images.githubusercontent.com/80331072/118098608-b1292800-b406-11eb-983f-e0eb43619a51.png)  

**有效实施**  
![image](https://user-images.githubusercontent.com/80331072/118099550-f863e880-b407-11eb-8172-d2b9f3119d88.png)

**C-SGD后的卷积核修剪**  
&emsp;&emsp;C-SGD训练完成后，由于每一簇的卷积核变得相同，所以挑选任何一个都是一样的。为每一层从剩下的集里简单地挑选每一簇中第一个卷积核（也就是有最小索引值的卷积核），就是  
![image](https://user-images.githubusercontent.com/80331072/118102031-d9b32100-b40a-11eb-9fff-469fb33342ae.png)  
&emsp;&emsp;对于下一层，我们把要删除的通道加到对应剩下的通道上。  
![image](https://user-images.githubusercontent.com/80331072/118102107-f2233b80-b40a-11eb-8919-9e0ac9ab50d3.png)  

## 核心代码
采用C-SGD方法的训练函数：
```
def train_one_step(net, data, label, optimizer, criterion, param_name_to_merge_matrix, param_name_to_decay_matrix):
    pred = net(data)
    loss = criterion(pred, label)
    loss.backward()

    #TODO note: C-SGD works here，核心部分
    for name, param in net.named_parameters():
        name = name.replace('module.', '')
        if name in param_name_to_merge_matrix:
            p_dim = param.dim()
            p_size = param.size()
            if p_dim == 4:
                param_mat = param.reshape(p_size[0], -1)
                g_mat = param.grad.reshape(p_size[0], -1)
            elif p_dim == 1:
                param_mat = param.reshape(p_size[0], 1)
                g_mat = param.grad.reshape(p_size[0], 1)
            else:
                assert p_dim == 2
                param_mat = param
                g_mat = param.grad
            #C-SGD的梯度跟新
            csgd_gradient = param_name_to_merge_matrix[name].matmul(g_mat) + param_name_to_decay_matrix[name].matmul(param_mat)
            param.grad.copy_(csgd_gradient.reshape(p_size))

    optimizer.step()
    optimizer.zero_grad()

    acc, acc5 = torch_accuracy(pred, label, (1,5))
    return acc, acc5, loss
```    
卷积核聚类函数：
```
def get_layer_idx_to_clusters(kernel_namedvalue_list, target_deps, pacesetter_dict):#
    result = {}
    for layer_idx, named_kv in enumerate(kernel_namedvalue_list):
        num_filters = named_kv.value.shape[0]
        if pacesetter_dict is not None and _is_follower(layer_idx, pacesetter_dict):
            continue
        if num_filters > target_deps[layer_idx]:
            result[layer_idx] = cluster_by_kmeans(kernel_value=named_kv.value, num_cluster=target_deps[layer_idx])#kmeans聚类，获取每个层的cluster
        elif num_filters < target_deps[layer_idx]:
            raise ValueError('wrong target dep')
    return result
```
剪枝函数：
```
csgd_prune_and_save(engine=engine, layer_idx_to_clusters=layer_idx_to_clusters,
                        save_file=pruned_weights, succeeding_strategy=succeeding_strategy, new_deps=target_deps)
```                        


## 代码运行(train on CIFAR-10)
1.Make a soft link to your CIFAR-10 directory.If the dataset is not found in the directory, it will be automatically downloaded
```
ln -s YOUR_PATH_TO_CIFAR cifar10_data
```
2.Set the environment variables.
```
export PYTHONPATH=.
```
3.Run Centripetal SGD to train a base ResNet-56
```
python csgd/do_csgd.py -a src56 -i 0
python csgd/do_csgd.py -a src56 -i 1
python csgd/do_csgd.py -a src56 -i 2
```
## 代码参考
[Centripetal-SGD](https://github.com/DingXiaoH/Centripetal-SGD0

