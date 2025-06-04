# 下面示例了一段可直接复制到你项目里的 Python 代码，用于在 MultiModalEncoder 上提取并可视化 “RGB→PC” 和 “PC→RGB” 两个方向的 Cross‐Attention 权重。
#
# 假设：
#   1. 你已经有一个训练好（或正在训练中）的 `encoder = MultiModalEncoder(...)` 实例。
#   2. 你拿到了一批样本 obs_dict，其中包含一个 RGB 图像张量 obs_dict["rgb"]，也有对应的点云张量 obs_dict["point_cloud"]。
#   3. 你还保留了原始的 RGB numpy 数组（H_img×W_img×3）和原始的点云 numpy 数组（N×3）用于可视化。
#   4. 在模型里，cross_attn 的 embed_dim = E，num_heads = H；图像侧投影后会得到 `[B, Q_rgb, E]`，点云侧投影后会得到 `[B, Q_pc, E]`。
#   5. 你知道图像特征映射（投影前）的空间大小，比如 (H_feat, W_feat)（常见做法是 ResNet18 最后一层 conv 输出）。
#
# 核心思路：
#   1. 用 encoder.img_proj 和 encoder.pc_proj 分别对图像特征和点云特征做线性投影，得到相同维度 E。
#   2. 调用同一个 MultiheadAttention，但要把 need_weights=True 才能拿到权重张量。
#   3. 输出的 attn_weights_rgb2pc 形状为 [B, H, Q_rgb, Q_pc]；attn_weights_pc2rgb 形状为 [B, H, Q_pc, Q_rgb]。
#   4. 调用前面示例过的可视化函数，把注意力权重叠到点云或图像上即可。
#
# 以下代码仅作示范，注意根据你自己项目中 `obs_dict` 的 key 名、特征维度、以及 H_feat, W_feat 调整参数。

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import open3d as o3d

# ------ 可视化函数：针对 RGB→PC ------
def visualize_rgb_to_pc_attention(attn_weights, point_cloud, head_idx=0, query_idx=None):
    """
    - attn_weights:  torch.Tensor([H, Q_rgb, Q_pc])
                     H = num_heads, Q_rgb = 图像投影后令牌数，Q_pc = 点云令牌数
    - point_cloud:   numpy.ndarray([Q_pc, 3])，原始点云坐标
    - head_idx:      要查看的注意力头索引 (0 ≤ head_idx < H)
    - query_idx:     如果指定，就可视化该图像令牌对所有点的注意力；否则对所有 query 平均.
    """
    hw_points = point_cloud.copy()  # [Q_pc, 3]
    hw_attn = attn_weights[head_idx].cpu().numpy()  # [Q_rgb, Q_pc]
    _, num_points = hw_attn.shape

    if query_idx is not None:
        point_importance = hw_attn[query_idx]   # [Q_pc]
    else:
        point_importance = hw_attn.mean(axis=0)  # 平均所有 query

    # 归一化到 [0,1]
    norm = Normalize(vmin=point_importance.min(), vmax=point_importance.max(), clip=True)
    colors = plt.cm.jet(norm(point_importance))[:, :3]  # [Q_pc, 3]

    # Open3D 可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hw_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

# ------ 可视化函数：针对 PC→RGB ------
def visualize_pc_to_rgb_attention(attn_weights, image, feat_map_size, head_idx=0, point_idx=None):
    """
    - attn_weights:   torch.Tensor([H, Q_pc, Q_rgb])
                      H = num_heads, Q_pc = 点云令牌数, Q_rgb = 图像投影后令牌数
    - image:          numpy.ndarray([H_img, W_img, 3])，原始 RGB 图像
    - feat_map_size:  (H_feat, W_feat) 图像投影后令牌的高、宽
    - head_idx:       选取要展示的注意力头
    - point_idx:      如果指定，就展示某个点对所有图像令牌的注意力；否则对所有点平均。
    """
    H_img, W_img, _ = image.shape
    H_feat, W_feat = feat_map_size
    hw_attn = attn_weights[head_idx].cpu().numpy()  # [Q_pc, Q_rgb]
    num_points, num_patches = hw_attn.shape

    if point_idx is not None:
        token_importance = hw_attn[point_idx]  # [Q_rgb]
    else:
        token_importance = hw_attn.mean(axis=0)  # 平均所有点

    # 从 [Q_rgb] 重塑到 [H_feat, W_feat]
    heatmap = token_importance.reshape(H_feat, W_feat)

    # 简单上采样到原图大小 (H_img, W_img)
    zoom_h = H_img // H_feat
    zoom_w = W_img // W_feat
    heatmap_resized = np.kron(heatmap, np.ones((zoom_h, zoom_w)))
    heatmap_resized = np.clip(heatmap_resized, 0, 1)

    # 画图：原图叠加热力图
    plt.figure(figsize=(6, 6))
    plt.imshow(image.astype(np.uint8))
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(f"PC→RGB Attention Head {head_idx}, PointIdx={point_idx}")
    plt.show()

# ------ 一次完整的提取 & 可视化 Pipeline ------
def visualize_cross_attention_for_one_sample(
    encoder: MultiModalEncoder,
    obs_dict: Dict[str, torch.Tensor],
    rgb_image_np: np.ndarray,
    point_cloud_np: np.ndarray,
    H_feat: int,
    W_feat: int,
    head_idx: int = 0,
    query_idx: int = None,
    point_idx: int = None,
    device: torch.device = None
):
    """
    1) 用 encoder 对 obs_dict 做一次前向，但不使用 model.forward()，而手动调用投影 + attention，
       以便获取 attn_weights_rgb2pc 与 attn_weights_pc2rgb。
    2) 再调用上面两个可视化函数分别画出注意力分布。
    参数说明：
    - encoder:         你的 MultiModalEncoder 实例。
    - obs_dict:        包含 "rgb"、"point_cloud" 等 key 的输入 dict (Tensor 已经在 device 上)。
    - rgb_image_np:    原始 RGB 图像 (H_img×W_img×3)，用于热力图叠加。
    - point_cloud_np:  原始点云 (N×3)，用于点云颜色上色。
    - H_feat, W_feat:  图像投影后令牌对应的高宽。比如 ResNet18 最后 conv 之后的特征图大小。
    - head_idx:        想要看的注意力头索引 (0 ≤ head_idx < num_heads)。
    - query_idx:       (可选) 指定某个图像令牌 index，用于 RGB→PC；如果是 None，则平均所有令牌。
    - point_idx:       (可选) 指定某个点云令牌 index，用于 PC→RGB；如果是 None，则平均所有点。
    - device:          torch.device，比如 torch.device("cuda:0")。
    """
    encoder.eval()
    if device is None:
        device = next(encoder.parameters()).device

    # —— 1. 提取 RGB 特征与 PointCloud 特征 —— #
    # 1.1 对 RGB 做 transform + backbone，取出最后一个 conv 后的特征 maps
    rgb_tensor = obs_dict["rgb"].to(device)  # [1, 3, H_img, W_img]
    # 假设 encoder.key_model_map["rgb"] 会输出形状 [1, C, H_feat, W_feat]
    with torch.no_grad():
        feat_map = encoder.key_model_map["rgb"](rgb_tensor)  # [1, C, H_feat, W_feat]
    B, C, Hf, Wf = feat_map.shape

    # 为了做 attention，需要把 [B, C, Hf, Wf] 展平成 [B, Q_rgb, C]，其中 Q_rgb = Hf*Wf
    feat_map_flat = feat_map.reshape(B, C, Hf*Wf).permute(0, 2, 1)  # [1, Q_rgb, C]

    # 对展平后的图像特征做线性投影到 embed_dim（encoder.img_proj）
    img_proj = encoder.img_proj(feat_map_flat)  # [1, Q_rgb, E]

    # 1.2 对点云做 transform + backbone，假设输出 [1, Q_pc, D_pc]
    pc_tensor = obs_dict["point_cloud"].to(device)  # 可能是 [1, N, 3] 或者 Dict 内部子键的组合
    with torch.no_grad():
        pc_feat = encoder.key_model_map["point_cloud"](pc_tensor)  # [1, Q_pc, D_pc]
    # 对点云特征做线性投影到 embed_dim
    pc_proj = encoder.pc_proj(pc_feat)  # [1, Q_pc, E]

    # 1.3 如果有低维状态 (low_dim)，也可在此提取并投影到同样 embed_dim，但此处示例仅用两模态
    #     low_dim_feat = encoder.key_model_map["low_dim"](obs_dict["low_dim"])  # [1, D_state]
    #     low_proj = encoder.state_proj(low_dim_feat.unsqueeze(1))  # [1, 1, E]

    # —— 2. 用投影后的 img_proj 和 pc_proj 调用 Cross‐Attention 并拿到权重 —— #
    # 注意：MultiheadAttention 在默认情况下需要输入形状为 [seq_len, B, E]；如果你在构造时用了 batch_first=True，
    #       则需要是 [B, seq_len, E]。上面代码假设使用 batch_first=True。
    # 确认下你的 encoder.cross_attn 是如何初始化的：如果是在 ._get_attention_func 中写了 `self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)`,
    # 那么默认是 batch_first=False，需要把张量先转成 [Q_rgb, B, E]、[Q_pc, B, E]。下面给出两种写法，具体选一个：

    # —— 情况 A： 如果你的 cross_attn 是用 batch_first=True 创建的，则直接：
    #     self.cross_attn = nn.MultiheadAttention(embed_dim=E, num_heads=H, batch_first=True)
    #     则 query/key/value 都要是 [B, seq_len, E]：
    img_q = img_proj  # [1, Q_rgb, E]
    pc_kv = pc_proj   # [1, Q_pc, E]
    # 调用时：
    attn_out_rgb2pc, attn_w_rgb2pc = encoder.cross_attn(
        query=img_q, key=pc_kv, value=pc_kv, need_weights=True
    )
    # attn_w_rgb2pc: [B, H, Q_rgb, Q_pc]

    # 同理 PC→RGB:
    pc_q = pc_proj    # [1, Q_pc, E]
    img_kv = img_proj # [1, Q_rgb, E]
    attn_out_pc2rgb, attn_w_pc2rgb = encoder.cross_attn(
        query=pc_q, key=img_kv, value=img_kv, need_weights=True
    )
    # attn_w_pc2rgb: [B, H, Q_pc, Q_rgb]

    # —— 情况 B： 如果你的 cross_attn 没有指定 batch_first=True，那么它期待 [seq_len, B, E] —— #
    #     # 转置维度：
    #     img_q_T = img_proj.permute(1, 0, 2)  # [Q_rgb, B, E]
    #     pc_kv_T = pc_proj.permute(1, 0, 2)   # [Q_pc, B, E]
    #     # RGB→PC:
    #     attn_out_rgb2pc, attn_w_rgb2pc = encoder.cross_attn(
    #         query=img_q_T, key=pc_kv_T, value=pc_kv_T, need_weights=True
    #     )
    #     # attn_w_rgb2pc: [B, H, Q_rgb, Q_pc]
    #
    #     # PC→RGB:
    #     attn_out_pc2rgb, attn_w_pc2rgb = encoder.cross_attn(
    #         query=pc_kv_T, key=img_q_T, value=img_q_T, need_weights=True
    #     )
    #     # attn_w_pc2rgb: [B, H, Q_pc, Q_rgb]
    #
    #     # 如果要进一步可视，需要把 attn_w_* 中的注意力权重取出来：
    #     # e.g. attn_w_rgb2pc = attn_w_rgb2pc.detach()  # [B, H, Q_rgb, Q_pc]
    #     #     attn_w_pc2rgb = attn_w_pc2rgb.detach()  # [B, H, Q_pc, Q_rgb]
    #
    #     # 由于下面示例直接用 A 情况，故注释了这部分。

    # —— 3. 开始可视化 —— #
    # 3.1 把 batch 维度取第 0 个样本，因为我们此时只看一张图/一个点云
    attn_w_rgb2pc = attn_w_rgb2pc[0]  # [H, Q_rgb, Q_pc]
    attn_w_pc2rgb = attn_w_pc2rgb[0]  # [H, Q_pc, Q_rgb]

    # 3.2 “RGB→PC” 注意力可视化
    visualize_rgb_to_pc_attention(
        attn_weights=attn_w_rgb2pc,
        point_cloud=point_cloud_np,
        head_idx=head_idx,
        query_idx=query_idx  # 可以传 None 让它平均所有 query
    )

    # 3.3 “PC→RGB” 注意力可视化
    visualize_pc_to_rgb_attention(
        attn_weights=attn_w_pc2rgb,
        image=rgb_image_np,
        feat_map_size=(H_feat, W_feat),
        head_idx=head_idx,
        point_idx=point_idx   # 可以传 None 让它平均所有点
    )

# -----------------------------------------------------------------------------
# 示例调用 (请修改 key 名、feat_map_size，根据你自己代码实际情况)：
#
# encoder = MultiModalEncoder(...)
# # 假设 obs_dict["rgb"] = torch.Tensor([1,3,H_img,W_img]), obs_dict["point_cloud"] = torch.Tensor([1,N,3])
# rgb_image_np = obs_dict["rgb"][0].permute(1,2,0).cpu().numpy()  # [H_img, W_img, 3]
# point_cloud_np = obs_dict["point_cloud"][0].cpu().numpy()      # [N, 3]
# H_feat, W_feat = 7, 7  # 如果 ResNet18 conv 最后输出 7×7
# visualize_cross_attention_for_one_sample(
#     encoder=encoder,
#     obs_dict=obs_dict,
#     rgb_image_np=rgb_image_np,
#     point_cloud_np=point_cloud_np,
#     H_feat=H_feat,
#     W_feat=W_feat,
#     head_idx=0,
#     query_idx=None,
#     point_idx=None,
# )
