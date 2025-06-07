from torch import tensor
state_stats = {
    "CloseBox":
    {
        'objects': {'box_base': {
            'dof_pos': {
                          'box_joint': (2.3635616302490234,
                                        2.3660526275634766,
                                        2.3641693592071533,
                                        0.0007281582220457494)},
                          'pos': (tensor([0.1686, 0.0390, 0.0747]), # min
                                  tensor([0.3765, 0.2970, 0.0747]), # max
                                  tensor([0.2805, 0.1651, 0.0747]), # mean
                                  tensor([0.0391, 0.0743, 0.0000])), # std
                          'rot': (tensor([ 0.6936, -0.1376, -0.7071, -0.1376]),
                                  tensor([ 0.7071,  0.0780, -0.6936,  0.0780]),
                                  tensor([ 0.7019, -0.0614, -0.7019, -0.0614]),
                                  tensor([0.0042, 0.0595, 0.0042, 0.0595]))}},
        'robots': {'franka': {'dof_pos': {
                                   'panda_finger_joint1': (0.03999445587396622,
                                                           0.04000464826822281,
                                                           0.03999971225857735,
                                                           2.0187230802548584e-06),
                                   'panda_finger_joint2': (0.03999524936079979,
                                                           0.040004123002290726,
                                                           0.04000011458992958,
                                                           1.879913611446682e-06),
                                   'panda_joint1': (-5.718467946280725e-06,
                                                    8.586646799813025e-06,
                                                    8.928792567530763e-07,
                                                    2.831436177075375e-06),
                                   'panda_joint2': (0.17492178082466125,
                                                    0.17536762356758118,
                                                    0.17518314719200134,
                                                    9.562807827023789e-05),
                                   'panda_joint3': (-2.4285644030896947e-05,
                                                    1.7914444470079616e-05,
                                                    -6.34520972653263e-07,
                                                    8.367778718820773e-06),
                                   'panda_joint4': (-0.8732666373252869,
                                                    -0.8729264140129089,
                                                    -0.8731129169464111,
                                                    7.217901293188334e-05),
                                   'panda_joint5': (-2.3955954020493664e-05,
                                                    1.9197810615878552e-05,
                                                    -1.5541439779553912e-06,
                                                    8.836418601276819e-06),
                                   'panda_joint6': (1.2214906215667725,
                                                    1.2216486930847168,
                                                    1.2215656042099,
                                                    3.377331086085178e-05),
                                   'panda_joint7': (0.7853797674179077,
                                                    0.7854176759719849,
                                                    0.7853990197181702,
                                                    6.977858447498875e-06)},
                       'pos': (tensor([-0.2677, -0.0053,  0.0003]),
                               tensor([-0.2677, -0.0053,  0.0003]),
                               tensor([-0.2677, -0.0053,  0.0003]),
                               tensor([0., 0., 0.])),
                       'rot': (tensor([ 9.9961e-01,  2.2156e-04,  2.8197e-03, -6.1979e-03]),
                               tensor([ 9.9961e-01,  2.2156e-04,  2.8197e-03, -6.1979e-03]),
                               tensor([ 9.9961e-01,  2.2156e-04,  2.8197e-03, -6.1979e-03]),
                               tensor([0., 0., 0., 0.]))}}}
}


import torch

def compute_stats(states):
    """
    输入：
        states: List[dict]，每个 dict 的结构相同，可以嵌套到任意深度，
                叶子节点要么是标量（int/float），要么是 torch.Tensor。
    返回：
        stats: dict，结构与 states[0] 相同，每个叶子节点被 (min, max, mean, std) 四元组替代。
    """
    def recurse(values_list):
        sample = values_list[0]
        # 如果是字典，就对每个 key 递归
        if isinstance(sample, dict):
            return {k: recurse([v[k] for v in values_list]) for k in sample}
        # 如果是标量
        elif isinstance(sample, (int, float)):
            arr = torch.tensor(values_list, dtype=torch.float32)
            return (arr.min().item(),
                    arr.max().item(),
                    arr.mean().item(),
                    arr.std(unbiased=False).item())
        # 如果是张量
        elif isinstance(sample, torch.Tensor):
            stacked = torch.stack(values_list, dim=0)   # 形状 [N, ...]
            mins  = stacked.min(dim=0).values           # 形状 [...]
            maxs  = stacked.max(dim=0).values
            means = stacked.mean(dim=0)
            stds  = stacked.std(dim=0, unbiased=False)
            return (mins, maxs, means, stds)
        else:
            raise TypeError(f"Unsupported leaf type: {type(sample)}")
    return recurse(states)



import random
import numpy as np
import torch

# 示例：state_stats 应当在别处初始化好，映射每个 task 到它的 statistics dict
# state_stats = {
#     'CloseBoxFrankaL0': { ... },  # 结构同 init_states[0]，叶子是 (min, max, mean, std)
#     ...
# }

def generate_random_states(state_dict, max_demos, seed=42):
    """
    根据 state_dict 中的 min/max/mean/std 生成 max_demos 个随机初始状态。

    Args:
        state_dict (dict): 嵌套结构，叶子节点是 (min, max, mean, std)。
        max_demos (int): 要生成的随机状态数量。

    Returns:
        List[dict]: 长度为 max_demos，每个元素结构与 state_dict 相同，
                    叶子节点替换为随机采样值（标量、ndarray 或 torch.Tensor）。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    def sample_once(stats):
        # 如果是子字典，则递归
        if isinstance(stats, dict):
            dic = {}
            for k, v in stats.items():
                if k == "rot":
                    dic[k] = sample_norm(v)
                else:
                    dic[k] = sample_once(v)
            return dic

        # 叶子节点：拆四元组
        min_val, max_val, mean_val, std_val = stats

        # 1) 标量分量
        if isinstance(mean_val, (int, float)):
            x = random.gauss(mean_val, std_val)
            return max(min_val, min(x, max_val))

        # 2) numpy 数组
        if isinstance(mean_val, np.ndarray):
            arr = np.random.normal(loc=mean_val, scale=std_val, size=mean_val.shape)
            return np.clip(arr, min_val, max_val)

        # 3) torch.Tensor
        if torch.is_tensor(mean_val):
            # PyTorch 的广播支持：mean_val/std_val 都可以是同形 tensor 或 scalar
            arr = torch.normal(mean=mean_val, std=std_val)
            arr = torch.min(arr, max_val)
            arr = torch.max(arr, min_val)
            return arr

        raise TypeError(f"Unsupported stats type: {type(mean_val)}")
    def sample_norm(stats):
        """专门处理 pos：按 mean/std 做高斯采样，再 clamp 到 [min, max]。"""
        min_v, max_v, mean_v, std_v = stats
        # 如果 mean_v 是 Tensor
        if torch.is_tensor(mean_v):
            x = torch.normal(mean=mean_v, std=std_v)
            if not (std_v == 0).all():
                x[2] = x[0]
                x[3] = x[1]
            return torch.clamp(x, min=min_v, max=max_v)
        raise TypeError(f"Unsupported norm stats type: {type(mean_v)}")

    # 为每一个 demo 采样一次
    return [sample_once(state_dict) for _ in range(max_demos)]



import matplotlib.pyplot as plt
from collections import defaultdict
def plot_state_distributions(states, bins=50):
    """
    对于一组 init_states，提取其中所有的 pos、rot、dof_pos 数值，
    并为每个名称 + 维度画出一个直方图。

    参数:
        states: list of dict, 每个元素格式同题主给出的 init_states
        bins: int, 直方图的柱子数
    """
    data = defaultdict(list)

    for st in states:
        # 先遍历 objects 和 robots 两个大类
        for category in ('objects', 'robots'):
            for name, ent in st[category].items():
                # 提取 pos, rot
                for key in ('pos', 'rot'):
                    if key in ent:
                        vals = ent[key]
                        for i, v in enumerate(vals):
                            data[f"{category}.{name}.{key}[{i}]"].append(v.item())
                # 提取每个 dof_pos
                for joint, v in ent.get('dof_pos', {}).items():
                    data[f"{category}.{name}.dof_pos.{joint}"].append(v)

    # 为每个维度绘制直方图
    for key, vals in data.items():
        plt.figure(figsize=(4,3))
        plt.hist(vals, bins=bins)
        plt.title(key)
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.tight_layout()

    plt.show()
