import subprocess
from enum import Enum

import torch


class LossType(Enum):
    ORIGINAL = 0
    L1_SPARSITY_REGULARIZATION = 1
    POLARIZATION = 4
    L2_POLARIZATION = 6

    @staticmethod#声明了静态方法。该方法不强制要求传递参数，可以实例化使用C().f(),也可以不实例化调用该方法C.f()；
    # classmethod 修饰符对应的函数不需要实例化，不需要 self 参数，但第一个参数需要是表示自身类的 cls 参数，可以来调用类的属性，类的方法，实例化对象等
    def from_string(desc: str):
        mapping = LossType.loss_name()
        return mapping[desc.lower()]

    @staticmethod
    def loss_name():
        return {"original": LossType.ORIGINAL,
                "sr": LossType.L1_SPARSITY_REGULARIZATION,
                "zol": LossType.POLARIZATION,
                "zol2": LossType.L2_POLARIZATION,
                }


def get_git_id() -> str:
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip().strip().decode()
    except subprocess.CalledProcessError:
        # the current directory is not a git repository
        return ""
    return commit_id


def compute_conv_flops(model: torch.nn.Module, cuda=False) -> float:
    """
    compute the FLOPs for CIFAR models
    NOTE: ONLY compute the FLOPs for Convolution layers and Linear layers
    """

    list_conv = []

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)

        flops = kernel_ops * output_channels * output_height * output_width

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        weight_ops = self.weight.nelement()

        flops = weight_ops
        list_linear.append(flops)

    def add_hooks(net, hook_handles: list):
        """
        apply FLOPs handles to conv layers recursively
        """
        children = list(net.children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                hook_handles.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                hook_handles.append(net.register_forward_hook(linear_hook))
            return
        for c in children:
            add_hooks(c, hook_handles)

    handles = []
    add_hooks(model, handles)
    demo_input = torch.rand(8, 3, 32, 32)
    if cuda:
        demo_input = demo_input.cuda()
        model = model.cuda()
    model(demo_input)

    total_flops = sum(list_conv) + sum(list_linear)

    # clear handles
    for h in handles:
        h.remove()
    return total_flops
