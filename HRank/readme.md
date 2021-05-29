# 剪枝-通道剪枝
## 下面要介绍的论文为：
[HRank: Filter Pruning using High-Rank Feature Map](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_HRank_Filter_Pruning_Using_High-Rank_Feature_Map_CVPR_2020_paper.pdf)(CVPR2020)
## 介绍
本篇提出基于feature map的秩的通道剪枝方法。该方法的依据有两点：  
(1)feature map作为中间变量，同时反映filter的属性和输入图像的信息。 即使在同一层内，单个feature map在网络中也扮演着不同的角色。并且feature map演示了输入图像每一层中的转换过程，最后转换为预测的标签。  
(2)作者经验观察到，filter输出的秩（即feature map的秩）的期望对输入图像具有鲁棒性。 可以证明，尽管不同的图像可能具有不同的秩，但方差可以忽略不计。   
![image](https://user-images.githubusercontent.com/80331072/120063638-c951a600-c09a-11eb-85b3-daaf77a749c6.png)  

## 方法
卷积核剪枝的目的是为了减去不重要的卷积核，本文对卷积核的重要性进行了重新定义，卷积核的重要性与输出特征图的rank有关。  
![image](https://user-images.githubusercontent.com/80331072/120064032-e38c8380-c09c-11eb-828b-7c3cf05d3a16.png)  
其中，输出特征图的rank可以经过SVD分解，得到高rank的特征和低rank的特征。因此，高阶特征图实际上比低阶特征图包含更多的信息。因此，rank排名可以作为一个可靠的衡量信息丰富度。
![image](https://user-images.githubusercontent.com/80331072/120064073-1a629980-c09d-11eb-82a2-75ff7a534f10.png)
为了准确地估计秩的期望，必须使用大量的输入图像。在评估卷积核的相对重要性时，这是一个巨大的挑战。本文通过实验观察，发现特征图的平均rank与Batch数量无关，因此可以使用较小的输入，去得到平均rank，也就是公式中的g=500张图片。  
![image](https://user-images.githubusercontent.com/80331072/120068457-61f42000-c0b3-11eb-9630-109870c269eb.png)  
**剪枝流程如下：**
1.使用输入计算出每个卷积核的输出特征图的平均Rank  
![image](https://user-images.githubusercontent.com/80331072/120068643-463d4980-c0b4-11eb-8d9c-514ded55f283.png)  
2.对得到的平均rank进行排序  
3.依据rank排序结果，决定剪枝的个数，然后对排序较后的卷积核进行剪枝  
4.以剩余的卷积核为初始参数进行finetune  
![image](https://user-images.githubusercontent.com/80331072/120068734-cebbea00-c0b4-11eb-92b0-1087d0ec8013.png)  

## 核心代码
获取每层的Rank：
```
handler = cov_layer.register_forward_hook(get_feature_hook_densenet)
```
```
def get_feature_hook_densenet(self, input, output):
    global feature_result
    global total
    a = output.shape[0]
    b = output.shape[1]
    c = torch.tensor([torch.matrix_rank(output[i,j,:,:]).item() for i in range(a) for j in range(b-12,b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    feature_result = feature_result / total
```
剪枝(掩码化）：
```
    def layer_mask(self, cov_id, resume=None, param_per_cov=4,  arch="vgg_16_bn"):
        params = self.model.parameters()
        prefix = "rank_conv/"+arch+"/rank_conv"
        subfix = ".npy"

        if resume:
            with open(resume, 'rb') as f:
                self.mask = pickle.load(f)
        else:
            resume=self.job_dir+'/mask'

        self.param_per_cov=param_per_cov

        for index, item in enumerate(params):

            if index == cov_id * param_per_cov:
                break
            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
                rank = np.load(prefix + str(cov_id) + subfix)
                pruned_num = int(self.compress_rate[cov_id - 1] * f)
                ind = np.argsort(rank)[pruned_num:]  # preserved filter id,排序获得保存下来的索引

                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                self.mask[index] = zeros  # covolutional weight
                item.data = item.data * self.mask[index]

            if index > (cov_id - 1) * param_per_cov and index <= (cov_id - 1) * param_per_cov + param_per_cov-1:
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

        with open(resume, "wb") as f:
            pickle.dump(self.mask, f)#将self.mask保存在f中
```            
## 代码运行
运行rank_generation获得Rank：(已提供训练好的Rank）  
```
python rank_generation.py
```
train and pruning:  
```
python main.py
```
## 代码参考
[lmbxmu/HRank](https://github.com/lmbxmu/HRank)
