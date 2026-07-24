# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """ 
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    我以后也能直接看源码吗？
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()

        # register_buffer()方法用于注册一个持久化的缓冲区，这个缓冲区不会被认为是模型的参数
        # 因此在调用model.parameters()时不会返回这个缓冲区
        # 是会存到model.state_dict()中的，但不会被优化器更新
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # 批量归一化当然是保留批量这个维度了

        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5  

        # 修改源码的那一步就在这里，在计算scale之前添加了一个小的常数eps，以避免除以零的情况

        # 为什么要用这样的方式去写呢？
        # 
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    # frozen Batch的含义就是在训练过程中，BatchNorm层的参数不更新，保持固定
    # 这意味着在训练过程中，BatchNorm层使用预先计算的均值和方差，而不是根据当前批次的数据进行计算
    # 这种方法可以减少训练过程中的不稳定性，特别是在小批量训练或分布式训练中
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        
        '''
        Input:
            name: str, backbone的名: 通常是ResNet的变体,如'resnet50'、'resnet101'等
            train_backbone: bool, 是否训练backbone的参数。如果为False,则backbone的参数将被冻结,不会在训练过程中更新。
            return_interm_layers: bool, 是否返回中间层的输出。如果为True,backbone将返回多个层的输出;如果为False,则只返回最后一层的输出。
            dilation: bool, 是否使用膨胀卷积。如果为True会替换ResNet最后两个阶段的部步长为膨胀率,增大感受野,常用于语义分割等任务
                                            如果为False,则使用标准卷积。     
        '''

        # 动态获取类并立即实例化
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],

            # is_main_process()的作用是检查当前进程是否是主进程
            # 在分布式训练中，通常会有多个进程同时运行，其中一个被指定为主进程
            # 只有主进程会执行某些操作，例如下载预训练模型或保存模型检查点，以避免重复的工作和资源浪费
            # 因此，is_main_process()函数用于确保只有主进程执行这些操作，而其他非主进程则跳过这些步骤。
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048

        # 让父类处理通用的初始化逻辑，子类只关注ResNet和冻结BN的定制化部分
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
