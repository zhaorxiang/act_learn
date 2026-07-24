import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        # args_override:之前参数超级字典的一部分
        super().__init__() # 初始化 PyTorch 内部的底层参数和钩子
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        # action:人类专家的目标关节角度序列。训练时必填，推理测试时为 None
        # is_pad:动作的填充掩码。训练时必填，测试时为 None

        # 让类像函数一样被执行
        env_state = None

        # ImageNet的均值和方差
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        
        # 在通道维度上进行标准化
        image = normalize(image)
        if actions is not None: # training time

            # 取出num_queries步的内容
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            # ..._hat就是对...的预测
            # a_hat：网络重构/预测出来的目标关节轨迹，维度是 [batch_size, 20, 14]。
            # is_pad_hat：预测的填充掩码（ACT 里主要看重构出的动作，此项不常用）。
            # mu：CVAE 隐空间分布的均值，维度是 [batch_size, 32]（隐变量维度是 32）。
            # logvar：CVAE 隐空间分布的对数方差，维度是 [batch_size, 32]
            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            
            # total_kld 代表 KL散度的大小
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            # 准备把损失函数留下
            loss_dict = dict()

            # 计算L1损失函数
            # reduction:就是保留维度，保留每一个时间步、每一个关节的误差值
            # all_l1的维度是[Batch_size, 20, 14]
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            # 总 Loss = L1 动作误差 + KL散度误差 × KL散度权重
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        # 将内部私有的优化器在外部留一个接口
        # 可直接通过 policy.optimizer 来调用

        # 支持更复杂的优化策略？不太理解。。。
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
