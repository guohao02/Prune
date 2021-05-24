# 剪枝——结构化剪枝
## 下面要介绍的论文为：
[Towards Optimal Structured CNN Pruning via Generative Adversarial Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Lin_Towards_Optimal_Structured_CNN_Pruning_via_Generative_Adversarial_Learning_CVPR_2019_paper.pdf)(CVPR 2019)  
## 介绍
这篇文章提出了非常新颖的结构剪枝策略：基于生成对抗的思路，将剪枝网络设置为生成器（Generator），其输出特征作为Fake，并设置Soft Mask门控一些异质结构的输出（如通道、分支、网络层或模块等）；将预训练模型设置为Baseline，Baseline的输出特征作为Real；再引入判别器（Discriminator）与正则化约束，一方面对齐生成器与Baseline的输出，另一方面驱使生成器中的Soft Mask稀疏化（mask value介于0到1之间），最终达到低精度损失的结构剪枝的目的。基于GAL（Generative Adversarial Learning）的剪枝策略总体如下图所示：  
![image](https://user-images.githubusercontent.com/80331072/119316695-d5f48980-bca9-11eb-9b2f-251143ebf765.png)  

## 方法
通过Soft Mask（标记为m）的稀疏化，可以剪除包括通道、分支或Block等在内的基本结构。为了确保剪枝之后，剪枝模型仍能获得与Baseline相接近的推理精度，基于GAL的剪枝策略首先对Soft Mask施加L1正则化；其次引入判别器（Discriminator），与剪枝模型（Generator）构成了生成对抗学习，在对抗学习过程中将Baseline输出的特征矢量作为监督信息，用以对齐Baseline与剪枝模型的特征输出。在对抗学习与正则化过程中，Baseline的参数固定、不需要更新，而剪枝模型参数WG、Soft Mask以及判别器参数WD需要更新，具体的优化问题如下：  
![image](https://user-images.githubusercontent.com/80331072/119317026-3edc0180-bcaa-11eb-8fd7-24bb05abff0c.png)  
**上式中的第一项表示表示判别器损失**，用来引导判别器提升鉴别能力，Baseline的输出表示Real，而剪枝模型（Generator）的输出表示Fake，当二者输出真假难辨时，达到对齐到输出特征的目的，其表达式如下：  
![image](https://user-images.githubusercontent.com/80331072/119317249-75198100-bcaa-11eb-86e3-0404e27a788e.png)  
**式(1)中的第二项为数据损失**，用来进一步对齐Baseline与Generator的输出特征，具体表示为Baseline与Generator输出特征之间的MSE损失，其表达式如下：  
![image](https://user-images.githubusercontent.com/80331072/119317526-c590de80-bcaa-11eb-9267-bfb3bdb96ee6.png)  
**式(1)的第三项为正则化损失**，主要分为三部分，分别表示对WG、m与WD的正则化约束，其内容如下所示：  
![image](https://user-images.githubusercontent.com/80331072/119317752-0852b680-bcab-11eb-9266-348aebe922b0.png)  
上式中R(WG)表示一般的weight decay，且通常是L2正则化；R(m)表示对Soft Mask的L1正则化；R(WD)表示对判别器的正则化约束，用以防止判别器主导训练学习，并且主要采用对抗正则化，促进判别器与生成器之间的对抗竞争，表达式如下：  
![image](https://user-images.githubusercontent.com/80331072/119317911-3df79f80-bcab-11eb-91f9-7a9ac1e8e3b6.png)  
**算法流程：**  
![image](https://user-images.githubusercontent.com/80331072/119318222-9dee4600-bcab-11eb-8836-975ac02a97b2.png)  
**优化策略**主要包含两个交替执行的阶段：  
**1）第一个阶段固定G与m**，通过对抗训练更新判别器D，损失函数包含对抗损失与对抗正则项  
**2）第二阶段固定D**，更新生成器G与Soft Mask，损失函数包含对抗损失中的fg相关项、fb与fg的MSE损失以及G、m的正则项。最终，完成Soft Mask的稀疏化之后，便可以按照门控方式，完成channel、branch或block的规整剪枝  
**第二阶段的m的更新**：文章采用FISTA 算法，如下图所示：  
![image](https://user-images.githubusercontent.com/80331072/119324022-edd00b80-bcb1-11eb-8ec8-a7b3fe5c2288.png) 

## 核心代码
优化器：
```
optimizer_d = optim.SGD(model_d.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)# 鉴别器优化器

param_s = [param for name, param in model_s.named_parameters() if 'mask' not in name]
param_m = [param for name, param in model_s.named_parameters() if 'mask' in name]

optimizer_s = optim.SGD(param_s, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)# 剪枝模型优化器
optimizer_m = FISTA(param_m, lr=args.lr, gamma=args.sparse_lambda)# mask的优化器
#step_size和gamma代表每走step_size个epoch，学习率衰减gamma倍，阶梯形式
scheduler_d = StepLR(optimizer_d, step_size=args.lr_decay_step, gamma=0.1)
scheduler_s = StepLR(optimizer_s, step_size=args.lr_decay_step, gamma=0.1)
scheduler_m = StepLR(optimizer_m, step_size=args.lr_decay_step, gamma=0.1)
```
train(两阶段):
```
############################
# (1) Update D network(Fix G and update D)
###########################

for p in model_d.parameters():  
    p.requires_grad = True  

optimizer_d.zero_grad()

output_t = model_d(features_t.detach())
#torch.full_like:将output_t的形状,real_label作为填充，返回结果tensor
labels_real = torch.full_like(output_t, real_label, device=args.gpus[0])
error_real = bce_logits(output_t, labels_real)#获取loss

output_s = model_d(features_s.to(args.gpus[0]).detach())

labels_fake = torch.full_like(output_t, fake_label, device=args.gpus[0])
error_fake = bce_logits(output_s, labels_fake)

error_d = error_real + error_fake

labels = torch.full_like(output_s, real_label, device=args.gpus[0])
error_d += bce_logits(output_s, labels)

error_d.backward()
losses_d.update(error_d.item(), inputs.size(0))
writer_train.add_scalar(
    'discriminator_loss', error_d.item(), num_iters)

optimizer_d.step()

if i % args.print_freq == 0:
    print(
        '=> D_Epoch[{0}]({1}/{2}):\t'
        'Loss_d {loss_d.val:.4f} ({loss_d.avg:.4f})\t'.format(
        epoch, i, num_iterations, loss_d=losses_d))

############################
# (2) Update student network
###########################

for p in model_d.parameters():  
    p.requires_grad = False  

optimizer_s.zero_grad()
optimizer_m.zero_grad()
#mse_loss():用于计算预测值和目标值的均方差误差,老师与学生的误差
error_data = args.miu * F.mse_loss(features_t, features_s.to(args.gpus[0]))

losses_data.update(error_data.item(), inputs.size(0))
writer_train.add_scalar(
    'data_loss', error_data.item(), num_iters)
error_data.backward(retain_graph=True)

# fool discriminator
output_s = model_d(features_s.to(args.gpus[0]))

labels = torch.full_like(output_s, real_label, device=args.gpus[0])
error_g = bce_logits(output_s, labels)#generator loss
losses_g.update(error_g.item(), inputs.size(0))
writer_train.add_scalar(
    'generator_loss', error_g.item(), num_iters)
error_g.backward(retain_graph=True)

# train mask
mask = []
for name, param in model_s.named_parameters():
    if 'mask' in name:
        mask.append(param.view(-1))
mask = torch.cat(mask)#torch.cat():在给定维度上对输入的张量序列seq 进行连接操作
# mask的l1
error_sparse = args.sparse_lambda * F.l1_loss(mask, torch.zeros(mask.size()).to(args.gpus[0]), reduction='sum')
error_sparse.backward()

losses_sparse.update(error_sparse.item(), inputs.size(0))
writer_train.add_scalar(
'sparse_loss', error_sparse.item(), num_iters)

optimizer_s.step()

decay = (epoch % args.lr_decay_step == 0 and i == 1)
if i % args.mask_step == 0:
    optimizer_m.step(decay)#fista算法

prec1, prec5 = utils.accuracy(features_s.to(args.gpus[0]), targets.to(args.gpus[0]), topk=(1, 5))
top1.update(prec1[0], inputs.size(0))
top5.update(prec5[0], inputs.size(0))
```
prune:
```
def prune_resnet(args, state_dict):
    thre = args.thre
    num_layers = int(args.student_model.split('_')[1])#层数
    n = (num_layers - 2) // 6
    layers = np.arange(0, 3*n ,n)
 
    mask_block = []
    for name, weight in state_dict.items():
        if 'mask' in name:
            mask_block.append(weight.item())

    pruned_num = sum(m <= thre for m in mask_block)#剪枝数
    pruned_blocks = [int(m) for m in np.argwhere(np.array(mask_block) <= thre)]#返回小于阈值的索引

    old_block = 0
    layer = 'layer1'
    layer_num = int(layer[-1])
    new_block = 0
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():# 将model_best的权值，根据mask去保留或者删除
        if 'layer' in key:
            if key.split('.')[0] != layer:
                layer = key.split('.')[0]
                layer_num = int(layer[-1])
                new_block = 0

            if key.split('.')[1] != old_block:
                old_block = key.split('.')[1]

            if mask_block[layers[layer_num-1] + int(old_block)] == 0:#如果对应的mask码为0
                if layer_num != 1 and old_block == '0' and 'mask' in key:
                    new_block = 1
                continue

            new_key = re.sub(r'\.\d+\.', '.{}.'.format(new_block), key, 1)
            if 'mask' in new_key: 
                new_block += 1

            new_state_dict[new_key] = state_dict[key]#数据传输，保留的block数据放在new_state_dict中

        else:
            new_state_dict[key] = state_dict[key]

    model = resnet_56_sparse(has_mask=mask_block).to(args.gpus[0])#模型修剪
```
## 代码运行：
prepare:
下载baseline：[resnet56](https://drive.google.com/file/d/1XHNxyFklGjvzNpTjzlkjpKc61-LLjt5T/view)  
train:  
```
python main.py
```
finetune:
```
python finetune.py --refine experiments/pruned.pt --pruned 
```
## 代码参考
[ShaohuiLin/GAL](https://github.com/ShaohuiLin/GAL)



