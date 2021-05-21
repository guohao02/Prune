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
## 算法流程
![image](https://user-images.githubusercontent.com/80331072/119096538-4c914d00-ba46-11eb-81d6-f81f13f35df5.png)  
## 核心代码
获取稀疏掩码
```
def get_sparse_mask():
            w_mask = apply_mask(weights, mask_init)
            logits = net.forward_pass(w_mask, self.inputs['input'],
                self.is_train, trainable=False)
            loss = tf.reduce_mean(compute_loss(self.inputs['label'], logits))
            grads = tf.gradients(loss, [mask_init[k] for k in prn_keys])#获得C(j)的梯度
            gradients = dict(zip(prn_keys, grads))
            cs = normalize_dict({k: tf.abs(v) for k, v in gradients.items()})
            return create_sparse_mask(cs, self.target_sparsity)
            
def create_sparse_mask(mask, target_sparsity):
    def threshold_vec(vec, target_sparsity):
        num_params = vec.shape.as_list()[0]
        kappa = int(round(num_params * (1. - target_sparsity)))
        topk, ind = tf.nn.top_k(vec, k=kappa, sorted=True)
        mask_sparse_v = tf.sparse_to_dense(ind, tf.shape(vec),
            tf.ones_like(ind, dtype=tf.float32), validate_indices=False)
        return mask_sparse_v
    if isinstance(mask, dict):
        mask_v, restore_fn = vectorize_dict(mask)
        mask_sparse_v = threshold_vec(mask_v, target_sparsity)
        return restore_fn(mask_sparse_v)
    else:
        return threshold_vec(mask, target_sparsity)      
```        
获取权值：
```
 # Switch for weights to use (before or after pruning)
 weights = tf.cond(self.pruned, lambda: net.weights_ap, lambda: net.weights_bp)
```
## 代码运行
python main.py
## 代码参考
[namhoonlee/snip-public](https://github.com/namhoonlee/snip-public)


