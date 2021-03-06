import torch


def hard_prune_network(network, args):#返回要剪枝的网络
    if args.network == 'vgg':
        network = hard_prune_vgg(network, args)
    elif args.network == 'resnet':
        network = hard_prune_resnet(network, args)
        # network = hard_prune_resnet_2(network, args)
    return network


def hard_prune_vgg(network, args):#调用hard_prune_vgg_step函数
    if network is None:
        return

    network = hard_prune_vgg_step(network, args.prune_layers, args.prune_channels, args.independent_prune_flag)

    print("-*-" * 10 + "\n\t\tPrune network\n" + "-*-" * 10)

    return network


def hard_prune_vgg_step(network, prune_layers, prune_channels, independent_prune_flag):
    count = 0  # count for indexing 'prune_channels'
    conv_count = 1  # conv count for 'indexing_prune_layers'
    dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]
    residue = None  # residue is need to prune by 'independent strategy'

    for i in range(len(network.features)):#len(network.features)为层数
        if isinstance(network.features[i], torch.nn.Conv2d):#当位为卷积层时
            if dim == 1:
                new_, residue = get_new_conv(network.features[i], dim, channel_index, independent_prune_flag)
                network.features[i] = new_
                dim ^= 1#异或

            if 'conv%d' % conv_count in prune_layers:
                channel_index = get_channel_index(network.features[i].weight.data, prune_channels[count], residue)#获取要剪枝的通道索引
                new_ = get_new_conv(network.features[i], dim, channel_index, independent_prune_flag)
                network.features[i] = new_
                dim ^= 1
                count += 1
            else:
                residue = None
            conv_count += 1
        elif dim == 1 and isinstance(network.features[i], torch.nn.BatchNorm2d):
            new_ = get_new_norm(network.features[i], channel_index)
            network.features[i] = new_

    if 'conv13' in prune_layers:#全连接层剪枝
        network.classifier[0] = get_new_linear(network.classifier[0], channel_index)

    return network


def hard_prune_resnet(network, args):
    if network is None:
        return

    # channel_index = get_channel_index(network.conv_1_3x3.weight.data, int(round(network.conv_1_3x3.out_channels * args.prune_rate[0])))
    # network.conv_1_3x3 = get_new_conv(network.conv_1_3x3, 0, channel_index, args.independent_prune_flag)
    # network.bn_1 = get_new_norm(network.bn_1, channel_index)

    for block in network.stage_1:
        block, _ = hard_prune_block(block, [], args.prune_rate[0], args.independent_prune_flag)
    for block in network.stage_2:
        block, _ = hard_prune_block(block, [], args.prune_rate[1], args.independent_prune_flag)
    for block in network.stage_3:
        block, _ = hard_prune_block(block, [], args.prune_rate[2], args.independent_prune_flag)

    # network.classifier = get_new_linear(network.classifier, channel_index)

    print("-*-" * 10 + "\n\t\tPrune network\n" + "-*-" * 10)

    return network


def hard_prune_block(block, channel_index, prune_rate, independent_prune_flag):
    # block.conv_a, greedy_residue = get_new_conv(block.conv_a, 1, channel_index, independent_prune_flag)
    channel_index = get_channel_index(block.conv_a.weight.data, int(round(block.conv_a.out_channels * prune_rate)), residue=None)
    block.conv_a = get_new_conv(block.conv_a, 0, channel_index, independent_prune_flag)
    block.bn_a = get_new_norm(block.bn_a, channel_index)

    block.conv_b, greedy_residue = get_new_conv(block.conv_b, 1, channel_index, independent_prune_flag)
    # channel_index = get_channel_index(block.conv_b.weight.data, int(round(block.conv_b.out_channels * prune_rate)), greedy_residue)
    # block.conv_b = get_new_conv(block.conv_b, 0, channel_index, independent_prune_flag)
    # block.bn_b = get_new_norm(block.bn_b, channel_index)

    return block, []


def hard_prune_resnet_2(network, args):
    if network is None:
        return

    channel_index = get_channel_index(network.conv_1_3x3.weight.data, int(round(network.conv_1_3x3.out_channels * args.prune_rate[0])))
    network.conv_1_3x3 = get_new_conv(network.conv_1_3x3, 0, channel_index, args.independent_prune_flag)
    network.bn_1 = get_new_norm(network.bn_1, channel_index)

    for block in network.stage_1:
        block, channel_index = hard_prune_block_2(block, channel_index, args.prune_rate[0], args.independent_prune_flag)
    for block in network.stage_2:
        block, channel_index = hard_prune_block_2(block, channel_index, args.prune_rate[1], args.independent_prune_flag)
    for block in network.stage_3:
        block, channel_index = hard_prune_block_2(block, channel_index, args.prune_rate[2], args.independent_prune_flag)

    network.classifier = get_new_linear(network.classifier, channel_index)

    print("-*-" * 10 + "\n\t\tPrune network\n" + "-*-" * 10)

    return network


def hard_prune_block_2(block, channel_index, prune_rate, independent_prune_flag):
    block.conv_a, greedy_residue = get_new_conv(block.conv_a, 1, channel_index, independent_prune_flag)
    channel_index = get_channel_index(block.conv_a.weight.data, int(round(block.conv_a.out_channels * prune_rate)), greedy_residue)
    block.conv_a = get_new_conv(block.conv_a, 0, channel_index, independent_prune_flag)
    block.bn_a = get_new_norm(block.bn_a, channel_index)

    block.conv_b, greedy_residue = get_new_conv(block.conv_b, 1, channel_index, independent_prune_flag)
    channel_index = get_channel_index(block.conv_b.weight.data, int(round(block.conv_b.out_channels * prune_rate)), greedy_residue)
    block.conv_b = get_new_conv(block.conv_b, 0, channel_index, independent_prune_flag)
    block.bn_b = get_new_norm(block.bn_b, channel_index)

    return block, channel_index


def get_channel_index(kernel, num_elimination, residue=None):#获取要剪枝通道的索引
    # get cadidate channel index for pruning
    # 'residue' is needed for pruning by 'independent strategy'

    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)#kernel.view(kernel.size(0), -1)将卷积核尺寸自适应转换为[通道数,X]，再绝对值按列相加
    if residue is not None:
        sum_of_kernel += torch.sum(torch.abs(residue.view(residue.size(0), -1)), dim=1)

    _, args = torch.sort(sum_of_kernel)#卷积核绝对值相加之和排序，并获取移动状况，即索引

    return args[:num_elimination].tolist()#转换为列表，前num_elimination个最小的索引


def index_remove(tensor, dim, index, removed=False):#更新张量
    if tensor.is_cuda:
        tensor = tensor.cpu()
    size_ = list(tensor.size())#列表化
    new_size = tensor.size(dim) - len(index)#通道数减去索引的长度，获得新的通道数
    size_[dim] = new_size#更新的通道数放在第二个维度中

    select_index = list(set(range(tensor.size(dim))) - set(index))#set()创建一个无序不重复元素集，可进行关系测试，删除重复数据
    new_tensor = torch.index_select(tensor, dim, torch.tensor(select_index))#dim:0表示按行索引，1表示按列索引，torch.index_select保留被索引的行或者列

    if removed:
        return new_tensor, torch.index_select(tensor, dim, torch.tensor(index))

    return new_tensor


def get_new_conv(conv, dim, channel_index, independent_prune_flag=False):#获得新的卷积层
    if dim == 0:
        new_conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                   out_channels=int(conv.out_channels - len(channel_index)),
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation, bias=conv.bias is not None)

        new_conv.weight.data = index_remove(conv.weight.data, dim, channel_index)#更新卷积层权值
        if conv.bias is not None:
            new_conv.bias.data = index_remove(conv.bias.data, dim, channel_index)#更新卷积层偏置

        return new_conv

    elif dim == 1:
        new_conv = torch.nn.Conv2d(in_channels=int(conv.in_channels - len(channel_index)),
                                   out_channels=conv.out_channels,
                                   kernel_size=conv.kernel_size,
                                   stride=conv.stride, padding=conv.padding, dilation=conv.dilation, bias=conv.bias is not None)

        new_weight = index_remove(conv.weight.data, dim, channel_index, independent_prune_flag)#更新权值
        residue = None
        if independent_prune_flag:
            new_weight, residue = new_weight
        new_conv.weight.data = new_weight
        if conv.bias is not None:
            new_conv.bias.data = conv.bias.data

        return new_conv, residue


def get_new_norm(norm, channel_index):#获得新的BN层
    new_norm = torch.nn.BatchNorm2d(num_features=int(norm.num_features - len(channel_index)),
                                    eps=norm.eps,
                                    momentum=norm.momentum,
                                    affine=norm.affine,
                                    track_running_stats=norm.track_running_stats)
    new_norm.weight.data = index_remove(norm.weight.data, 0, channel_index)#更新BN层权值
    new_norm.bias.data = index_remove(norm.bias.data, 0, channel_index)#更新BN层偏置

    if norm.track_running_stats:
        new_norm.running_mean.data = index_remove(norm.running_mean.data, 0, channel_index)
        new_norm.running_var.data = index_remove(norm.running_var.data, 0, channel_index)

    return new_norm


def get_new_linear(linear, channel_index):#获取新的全连接层
    new_linear = torch.nn.Linear(in_features=int(linear.in_features - len(channel_index)),
                                 out_features=linear.out_features,
                                 bias=linear.bias is not None)
    new_linear.weight.data = index_remove(linear.weight.data, 1, channel_index)#更新全连接层的权值
    new_linear.bias.data = linear.bias.data#更新全连接层的偏置

    return new_linear
