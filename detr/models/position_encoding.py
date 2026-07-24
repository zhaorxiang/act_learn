# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor

import IPython
e = IPython.embed

class PositionEmbeddingSine(nn.Module):

# num_pos_feat(number of position feats):位置特征的数量
# dim_t(dimension temperature):维度频率调节分母

    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        # scale=None:坐标缩放的最大上限，默认是空值
        
        super().__init__() # 调用父类初始化方法
        self.num_pos_feats = num_pos_feats # 传入的通道数
        self.temperature = temperature # 正弦波的频率调节温度常数，默认是10000
        self.normalize = normalize # 是否开启坐标归一化，默认是关闭

        # 想缩放却关闭了归一化
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi # 默认的2 * pi
        self.scale = scale

    def forward(self, tensor):
        # tensor:输入的图像特征张量，维度是 [batch_size, channels, height, width]
        x = tensor
        # mask = tensor_list.mask
        # assert mask is not None
        # not_mask = ~mask

        not_mask = torch.ones_like(x[0, [0]])

        # axis=1: 沿着高度维度进行累积求和，得到y_embed，每一行的值都是相等的
        # axis=2: 沿着宽度维度进行累积求和，得到x_embed，每一列的值都是相等的
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6

            # y_embed[:, -1:, :]的维度是[batch_size, 1, width]
            # 结果正好是每一列的最大值，每一列除以这个之后就归一化到0-1之间了
            # x_embed也是类似的操作，除以每一行的最大值归一化到0-1之间
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # pos_x的维度是 [batch_size, height, width, num_pos_feats]
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        # 保证sin和cos相互交替
        # 机制:首先是扩充一个维度，所以倒数第二个维度会对应sin和cos两个数据
        # 再进行flatten(3)操作，把倒数第二个维度和最后一个维度合并成一个维度，这样就保证了sin和cos相互交替
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
