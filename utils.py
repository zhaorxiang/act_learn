import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats

        # Is Simulation?
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        # 不读取完整的400步，而是后面随机切片
        sample_full_episode = False # hardcode

        # 拿出第index个文件
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            
            # 读取是不是仿真数据，这个标签是在录制仿真脚本的时候搞上去的
            is_sim = root.attrs['sim']

            # 读取HDF5里的action大小，维度是[400, 14]
            original_action_shape = root['/action'].shape
            # 获取这一集录像的总长度400帧
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                # 在0~399之间随机抽取一个数字
                start_ts = np.random.choice(episode_len)

            # get observation at start_ts only
            # 提取start_ts这一秒的状态,维度:[14]
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]

            # 用来装图像的空字典
            image_dict = dict()
            for cam_name in self.camera_names:
                # 图像的维度是:[480, 640, 3]
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                # 直接截取start_ts到399步的所有动作
                action = root['/action'][start_ts:]
                # 计算剩下真实动作步数长度
                action_len = episode_len - start_ts
            else:
                # 如果是真机世界
                # 动作起点往前挪一帧，对齐真机电机的网络物理延迟
                # 保证没有从负数开始
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        # padded_action维度:[400, 14]
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        # 用在自注意力机制中无视这部分
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)
        # 堆叠后：维度变成[4(摄像头个数), 480, 640, 3]

        # construct observations
        # numpy数据转换成能用的Tensor张量
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()# [400, 14]
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        # 通道转换:[4, 3, 480, 640]
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        # 图像归一化
        image_data = image_data / 255.0 # [0, 1]
        # 归一化
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        # 归一化
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # 在 Transformer 的自注意力计算中，我们不希望网络去学习那些无意义的“零”
        # is_pad 就会告诉 Transformer：“注意！后面标了 True 的步骤全是我补的零，
        # 你在计算时请自动忽略它们
        # 错误：我的理解是训练数据实际上是16维度的，但是只有14维度有意义所以要用这个？

        # 正确的理解：实际上是比如去20步，但是我再有5步就完成了，剩下的步骤就是0了
        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    '''
    Input:
        dataset_dir:存放hdf5文件的路径
        num_episodes:演示视频数

    Return:
        stats:一个大字典{包含所有原始数据的均值和标准差，以及关节信息}
    '''

    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            # Joint Position
            qpos = root['/observations/qpos'][()]
            # Joint velocities
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

    # 在第 0 维强行增加一个维度，把列表里的张量“叠”成一个高维大张量    
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)

    # 无用的废话或者打断点算维度使用
    all_action_data = all_action_data

    #维度是[演示视频个数, 每一个视频的帧数, 关节角度维度]


    # 沿着第0维和第1维求平均，然后求出的每一个关节/动作的均值 [1, 1, 14]

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)

    # 把标准差限制在 0.01（1e-2）到无穷大（np.inf）之间
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping
    
    # .squeeze()：去掉所有大小为 1 的无用维度
    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    '''
    Input:
        dataset_dir:存放hdf5的文件夹路径
        num_episodes:演示视频
        camera_names:摄像机名称列表,传入的是一个List
        batch_size_train:训练批次大小
        batch_size_val:验证批次大小

    Return:
        train_dataloader:训练数据
        val_dataloader:验证数据
        norm_stats:使用所有数据计算的用来归一化的均值和方差
        train_dataset.is_sim:是否是仿真环境
    '''


    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8 # 80%训练集
    # 随机打乱,然后取前80%作为训练集,后20%作为测试集,这样就做到了随机选择的效果
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
