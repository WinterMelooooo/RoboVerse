import numpy as np
import math
import os
import torch
from curobo.types.math import Pose
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
import pdb
import sys
import copy
import random
import json
from tqdm import tqdm

FR3_URDF = "/home/ghr/yktang/RoboVerse/roboverse_data/robots/franka/urdf/franka_panda.urdf"
BASE_LINK = "panda_link0"
EE_LINK = "panda_hand"

EXPECTED_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
]


def init_ik_solver(urdf_file=FR3_URDF, base_link=BASE_LINK, ee_link=EE_LINK, use_cuda_graph=False):
    tensor_args = TensorDeviceType()
    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=use_cuda_graph,
    )
    return IKSolver(ik_config), tensor_args


def _to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    x = np.asarray(x)
    return torch.from_numpy(x).to(device=device, dtype=dtype)


def _quat_mul_xyzw(q1, q2):
    # q = q1 * q2, xyzw
    x1, y1, z1, w1 = q1.unbind(-1)
    x2, y2, z2, w2 = q2.unbind(-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def _choose_grid_n(std: float, max_spacing: float = 0.05, n: int | None = None) -> int:
    """
    选择每个被启用轴向的网格分辨率 n：
    - 至少 3
    - 间距 <= max_spacing
    - n 为奇数（保证包含 0）
    - 若传入 n，则在此基础上向上调整以满足上述约束
    """
    if std <= 0:
        return 1  # 该轴不随机或无范围：仅中心点

    if n is None or n < 1:
        # 根据间距约束推最小 n： 2*std/(n-1) <= max_spacing
        n_req = int(np.ceil(2.0 * float(std) / float(max_spacing))) + 1
        n = max(3, n_req)
    else:
        n = max(3, int(n))

    # 保证奇数（包含中心点0）
    if n % 2 == 0:
        n += 1

    # 再次确保间距约束
    spacing = (2.0 * float(std)) / (n - 1)
    if spacing > max_spacing:
        # 需要再加大 n，直到满足 spacing<=max_spacing，且保持奇数
        n_req = int(np.ceil(2.0 * float(std) / float(max_spacing))) + 1
        if n_req % 2 == 0:
            n_req += 1
        n = max(n, n_req)

    return n


def _grid_masked(std: float, along_xyz, n: int | None = None, max_spacing: float = 0.05) -> torch.Tensor:
    """
    在被启用的轴向上，以 [-std, +std] 的闭区间均匀生成网格（linspace），
    并从笛卡尔积格点中“均匀随机”选择一个点作为偏移。
    - 未启用的轴向（along=0）偏移恒为 0
    - n 至少 3，奇数，且相邻间距 <= max_spacing
    - 中心点(0)必在网格中
    返回: shape [3] 的 torch.float32 向量
    """
    mask = torch.tensor(along_xyz, dtype=torch.float32)
    std = float(std)

    # 为每个轴决定 n_i
    def axis_linspace(on_axis: bool) -> np.ndarray:
        if not on_axis or std <= 0:
            return np.array([0.0], dtype=np.float32)
        n_i = _choose_grid_n(std=std, max_spacing=max_spacing, n=n)
        return np.linspace(-std, std, n_i, dtype=np.float32)

    xs = axis_linspace(bool(mask[0].item()))
    ys = axis_linspace(bool(mask[1].item()))
    zs = axis_linspace(bool(mask[2].item()))

    # 笛卡尔积格点数量
    total = len(xs) * len(ys) * len(zs)
    # 均匀随机取一个索引
    idx = np.random.randint(0, total)

    # 将线性索引展开为 (ix, iy, iz)
    iz = idx % len(zs)
    iy = (idx // len(zs)) % len(ys)
    ix = idx // (len(ys) * len(zs))

    pt = np.array([xs[ix], ys[iy], zs[iz]], dtype=np.float32)
    # 未启用轴向归零（保险）
    pt = pt * mask.numpy()
    return torch.tensor(pt, dtype=torch.float32)


def _apply_post_grasp_lift(traj, first_span_end, ik_solver, device,
                           lift=0.20, smooth_frames=10, joint_names=EXPECTED_JOINT_NAMES):
    """
    在第一段（抓取）结束后的时刻 first_span_end 之后，做抬高动作：
    - 取 t = first_span_end
    - 以 t+K 帧 (K=smooth_frames) 的末端位姿为基准，Z 方向抬高 lift
    - IK 得到 q_lift，与 q_{t+K} 做差 Δq
    - 对 (t, t+K] 做线性平滑加 Δq；对 (t+K, 末尾] 全量加 Δq
    """
    T = len(traj)
    if T == 0:
        return True  # 无操作，视为成功

    t = int(first_span_end)
    if t >= T - 1:
        return True  # 没有后续帧，无需抬高

    # 实际可用的平滑帧数
    K = min(int(smooth_frames), max(1, T - 1 - t))
    tK = t + K

    # 取 q_{t+K}
    q_tK = torch.tensor([traj[tK]['franka']['dof_pos_target'][jn] for jn in joint_names],
                        dtype=torch.float32, device=device).view(1, -1)

    # FK 得到末端位姿
    fk_out = ik_solver.fk(q_tK)  # 期望返回 ee_position [1,3], ee_quaternion [1,4]
    ee_pos = fk_out.ee_position
    ee_quat = fk_out.ee_quaternion

    # 目标位姿：抬高 Z
    target_pos = ee_pos.clone()
    target_pos[:, 2] = target_pos[:, 2] + float(lift)
    target_quat = ee_quat  # 姿态不变
    target_pose = Pose(target_pos, target_quat)

    # IK
    ik_res = ik_solver.solve_batch(target_pose)
    ok = bool(ik_res.success[0].item()) if hasattr(ik_res, "success") else True
    count = 1
    max_count = 5
    while not ok:
        ik_res = ik_solver.solve_batch(target_pose)
        ok = bool(ik_res.success[0].item()) if hasattr(ik_res, "success") else True
        count += 1
        if count > max_count:
            break
    if not ok:
        #print(f"[lift] IK failed at t+K (t={t}, K={K}). Skip lifting.")
        return False

    q_lift = ik_res.solution.squeeze(0)  # [1,7] -> [7]
    if q_lift.ndim == 2:
        q_lift = q_lift[0]
    if q_lift.numel() != len(joint_names):
        print(f"[lift] IK solution shape mismatch: {tuple(q_lift.shape)}")
        return False

    # Δq
    dq = (q_lift - q_tK.view(-1)).detach().cpu().numpy()

    # 线性平滑： (t, t+K] 加 α * Δq
    for i in range(1, K + 1):
        alpha = i / float(K)
        frame = traj[t + i]['franka']['dof_pos_target']
        for j, jn in enumerate(joint_names):
            frame[jn] = float(frame[jn] + alpha * dq[j])

    # 之后全量加 Δq： (t+K, T)
    for tt in range(tK + 1, T):
        frame = traj[tt]['franka']['dof_pos_target']
        for j, jn in enumerate(joint_names):
            frame[jn] = float(frame[jn] + dq[j])

    return True

def euler_to_quaternion_xyz(roll, pitch, yaw):
    cr = np.cos(roll / 2.0); sr = np.sin(roll / 2.0)
    cp = np.cos(pitch / 2.0); sp = np.sin(pitch / 2.0)
    cy = np.cos(yaw / 2.0);   sy = np.sin(yaw / 2.0)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    q = np.array([x, y, z, w], dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-9)
    return q



def save_init_states(init_states, filename):
    """
    将 init_states 存储为 JSON 文件
    - 自动把 torch.Tensor / numpy.ndarray 转为 list
    - 其他类型保持不变
    """
    def to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        else:
            return obj

    serializable_states = [to_serializable(state) for state in init_states]

    with open(filename, "w") as f:
        json.dump(serializable_states, f, indent=2)

    print(f"✅ Saved init_states to {filename}")

def _get_joint_vec_from_frame(frame_dict, joint_names):
    return [frame_dict['franka']['dof_pos_target'][jn] for jn in joint_names]


def update_traj(single_traj, delta_pos_ee, delta_rot_ee, start_idx, end_idx, end_gripper_idx,
                ik_solver, device, dtype=torch.float32):
    """
    single_traj: [timesteps, dict]
    delta_pos_ee: [3,]
    delta_rot_ee: [4,] (xyzw), new = origin * delta
    """
    s = int(start_idx); e = int(end_idx)
    T = len(single_traj)
    if not (0 <= s < T) or not (0 <= e < T):
        raise IndexError(f"start/end 超界: len={T}, s={s}, e={e}")
    if e < s:
        # 没有有效区间，直接返回
        return single_traj

    # 起点/终点关节
    q_s_list = _get_joint_vec_from_frame(single_traj[s], EXPECTED_JOINT_NAMES)
    q_e_list = _get_joint_vec_from_frame(single_traj[e], EXPECTED_JOINT_NAMES)
    q_s = _to_tensor(q_s_list, device=device, dtype=dtype).view(1, -1)  # [1,7]
    q_e = _to_tensor(q_e_list, device=device, dtype=dtype).view(1, -1)  # [1,7]

    # 用“终点 e 帧”的末端位姿加偏置得到目标末端
    fk_out = ik_solver.fk(q_e)
    ee_pos = fk_out.ee_position       # [1,3]
    ee_quat = fk_out.ee_quaternion    # [1,4] (xyzw)

    dpos = _to_tensor(delta_pos_ee, device=device, dtype=dtype).view(1, 3)
    drot = _to_tensor(delta_rot_ee, device=device, dtype=dtype).view(1, 4)
    new_pos = ee_pos + dpos
    new_quat = _quat_mul_xyzw(ee_quat, drot)
    new_quat = new_quat / (new_quat.norm(dim=-1, keepdim=True) + 1e-9)

    target_pose = Pose(new_pos, new_quat)

    # IK
    ik_res = ik_solver.solve_batch(target_pose)
    ok = bool(ik_res.success[0].item()) if hasattr(ik_res, "success") else True
    if not ok:
        return None

    q_new = ik_res.solution.squeeze(0)  # [1,7]
    if q_new.ndim != 2 or q_new.shape[1] != len(EXPECTED_JOINT_NAMES):
        raise ValueError(f"IK 返回形状异常：{q_new.shape}（期望 [1,{len(EXPECTED_JOINT_NAMES)}]）")
    q_new = q_new[0]  # [7]
    q_s = q_s[0]      # [7]

    # 关节插值：q(s) -> q_new，覆盖 [s..e]
    steps = e - s
    if steps == 0:
        q_t = q_new
        frame = single_traj[s]['franka']['dof_pos_target']
        for i, jn in enumerate(EXPECTED_JOINT_NAMES):
            frame[jn] = float(q_t[i].item())
        return single_traj

    for t in range(s, e + 1):
        alpha = (t - s) / float(steps)  # t=s → 0, t=e → 1
        q_t = (1 - alpha) * q_s + alpha * q_new
        frame = single_traj[t]['franka']['dof_pos_target']
        for i, jn in enumerate(EXPECTED_JOINT_NAMES):
            frame[jn] = float(q_t[i].item())

    end_frame = single_traj[e]['franka']['dof_pos_target']
    for t in range(e+1, end_gripper_idx + 1):
        frame = single_traj[t]['franka']['dof_pos_target']
        for i, jn in enumerate(EXPECTED_JOINT_NAMES):
            frame[jn] = end_frame[jn]

    return single_traj


def split_traj(all_actions, threshold=0.005):
    """
    返回每条轨迹的“非抓/放”连续段 [start, end]
    """
    # 夹爪关节名容错
    cand_keys = ["panda_finger_joint1", "panda_finger_joint_1"]
    split_idxs_all = []

    for traj in all_actions:
        # 找到实际存在的 key
        key = None
        for k in cand_keys:
            if k in traj[0]["franka"]["dof_pos_target"]:
                key = k
                break
        if key is None:
            raise KeyError(f"未找到 finger joint 键，尝试过 {cand_keys}")

        vals = []
        for step in traj:
            v = step["franka"]["dof_pos_target"][key]
            v = float(v.item()) if hasattr(v, "item") else float(v)
            vals.append(v)

        T = len(vals)
        if T == 0:
            split_idxs_all.append([])
            continue

        grasping = [False] * T
        for t in range(1, T):
            if abs(vals[t] - vals[t - 1]) > threshold:
                grasping[t] = True

        spans = []
        in_span = False
        start = 0
        for t in range(T):
            if not grasping[t]:
                if not in_span:
                    in_span = True
                    start = t
            else:
                if in_span:
                    spans.append([start, t - 1])
                    in_span = False
        if in_span:
            spans.append([start, T - 1])

        spans = [s for s in spans if s[0] <= s[1]]
        split_idxs_all.append(spans)

    return split_idxs_all

def _to_np_array(v):
    try:
        import torch
        if isinstance(v, torch.Tensor):
            return v.detach().cpu().float().numpy()
    except Exception:
        pass
    return np.asarray(v, dtype=np.float32)

def plot_object_position_distributions(
    init_states,
    save_path,
    plane: str = "xy",     # 可选: "xy" | "xz" | "yz"
    bins: int = 50,        # hexbin/bin 密度
    min_points_scatter: int = 100,  # 少于该阈值用散点
    figsize_per_subplot: float = 4.0  # 单子图边长
):
    """
    汇总所有 init_states 中每个 object 的位置分布，并在同一张图里分别作图后保存。
    - init_states: List[dict]，其中每个元素应含 "objects" 字段，每个 object 有 "pos" (shape: 3,)
    - save_path: 输出图片路径，例如 "/path/to/obj_pos_dist.png"
    - plane: 选择作图平面: "xy" | "xz" | "yz"
    """
    # 选择维度
    plane = plane.lower()
    if plane not in ("xy", "xz", "yz"):
        raise ValueError(f"plane 必须是 'xy'/'xz'/'yz' 之一，收到: {plane}")
    idx_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    ax_i, ax_j = idx_map[plane]
    axis_labels = {0: "x (m)", 1: "y (m)", 2: "z (m)"}
    x_label, y_label = axis_labels[ax_i], axis_labels[ax_j]

    # 汇总每个 object 的位置
    obj2pos = {}  # name -> list of (3,)
    for st in init_states:
        objs = st.get("objects", {})
        for name, od in objs.items():
            pos = od.get("pos", None)
            if pos is None:
                continue
            p = _to_np_array(pos).reshape(-1)  # (3,)
            if p.size < 3:
                continue
            obj2pos.setdefault(name, []).append(p)

    if not obj2pos:
        raise ValueError("未在 init_states 中找到任何含 'pos' 的 objects。")

    # 排序以便固定子图顺序
    obj_names = sorted(obj2pos.keys())
    n = len(obj_names)

    # 画布网格：最多 3 列，行数自适应
    cols = min(3, n)
    rows = math.ceil(n / cols)

    # 非交互式后端（适合服务器）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_w = cols * figsize_per_subplot
    fig_h = rows * figsize_per_subplot
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    if n == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    # 逐 object 绘图
    for k, name in enumerate(obj_names):
        r, c = divmod(k, cols)
        ax = axes[r, c]

        data = np.stack(obj2pos[name], axis=0)  # (M, 3)
        xs = data[:, ax_i]
        ys = data[:, ax_j]

        # 选择散点 or hexbin
        if data.shape[0] < min_points_scatter:
            ax.plot(xs, ys, ".", markersize=2, alpha=0.8)
        else:
            hb = ax.hexbin(xs, ys, gridsize=bins)
            cb = fig.colorbar(hb, ax=ax)
            cb.set_label("count")

        # 轴标签、等比例与标题
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect("equal", adjustable="box")

        # 坐标范围 & 标题附带统计
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        ax.set_title(
            f"{name}  (N={len(xs)})\n"
            f"{plane}: x∈[{x_min:.3f},{x_max:.3f}], y∈[{y_min:.3f},{y_max:.3f}]"
        )

    # 清理空子图（当 n 不是 rows*cols 时）
    total_slots = rows * cols
    if total_slots > n:
        for k in range(n, total_slots):
            r, c = divmod(k, cols)
            axes[r, c].axis("off")
    save_path = os.path.join(save_path, "init_states.png")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Save plot to: {save_path}")

def repeat_on_rand_states(init_states, all_actions, all_states, delta_pos_butter, delta_rot_quat_butter, delta_pos_basket, delta_rot_quat_basket, split_idxs_list, num_repeat_trajs_for_single_state):
    new_init_states = []
    new_all_actions = []
    new_all_states = []
    new_delta_pos_butter = []
    new_delta_rot_quat_butter = []
    new_delta_pos_basket = []
    new_delta_rot_quat_basket = []
    new_split_idxs_list = []
    num_original_states = len(init_states)
    num_original_trajs = len(all_actions)
    print(f"Before scaling, we have {num_original_states} states and {num_original_trajs} actions.")
    for i,state in enumerate(init_states):
        for _ in range(num_repeat_trajs_for_single_state):
            new_init_states.append(state)
            new_delta_pos_butter.append(delta_pos_butter[i])
            new_delta_rot_quat_butter.append(delta_rot_quat_butter[i])
            new_delta_pos_basket.append(delta_pos_basket[i])
            new_delta_rot_quat_basket.append(delta_rot_quat_basket[i])
            idx = random.randint(0,num_original_trajs-1)
            new_all_actions.append(all_actions[idx])
            new_all_states.append(all_states[idx])
            new_split_idxs_list.append(split_idxs_list[idx])
    print(f"After scaling, we have {len(new_init_states)} states and {len(new_all_actions)} actions.")
    return new_init_states, new_all_actions, new_all_states, new_delta_pos_butter, new_delta_rot_quat_butter, new_delta_pos_basket, new_delta_rot_quat_basket, new_split_idxs_list

def randomize_objects_and_traj(
    init_states, all_actions, all_states, randomization_cfg,
    urdf_file=FR3_URDF, base_link=BASE_LINK, ee_link=EE_LINK,
    min_sep=0.05, max_retries=50,
    save_init_states_dir="/home/ghr/yktang/RoboVerse/tmp/init_states",
    task_name="LiberoPickButter",
    rand_traj=True,
    plot_dir=None,
    num_trajs_per_state=4
):
    """
    - 对 init_states 中指定 objects 随机化 (pos/quat)
    - 约束：任意两个物体中心距离 >= min_sep，否则重采（最多 max_retries 次）
    - 将轨迹的两个“非抓/放”片段分别按 butter / basket 的 delta 做 IK 修正
    - 对于任一段 IK 失败的 demo：**直接跳过**（不进入返回结果）
    返回：(filtered_init_states, filtered_updated_actions, filtered_all_states)
    """
    try:
        # 1) 计算“非抓/放”片段
        split_idxs_list = split_traj(all_actions)

        # 2) 为每条轨迹准备 delta（并在 state 上回写随机后的 pos/quat）
        delta_pos_ee_butter = []
        delta_rot_quat_butter = []
        delta_pos_ee_basket = []
        delta_rot_quat_basket = []


        def _uniform_masked(std, along_xyz):
            lo = -float(std); hi = float(std)
            mask = torch.tensor(along_xyz, dtype=torch.float32)
            return (lo + (hi - lo) * torch.rand(3)) * mask

        def _to_tensor_pos(x):
            if torch.is_tensor(x):
                return x.clone().float()
            return torch.tensor(x, dtype=torch.float32)

        def _to_tensor_quat(x):
            if torch.is_tensor(x):
                return x.clone().float()
            return torch.tensor(x, dtype=torch.float32)

        def _quat_right_mul(q0_xyzw, dq_xyzw):
            x1, y1, z1, w1 = np.asarray(q0_xyzw, dtype=np.float32)
            x2, y2, z2, w2 = np.asarray(dq_xyzw, dtype=np.float32)
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            q = np.array([x, y, z, w], dtype=np.float32)
            q /= (np.linalg.norm(q) + 1e-9)
            return q

        def _sample_once_and_apply(state, rnd_cfg):
            updated_objs, deltas = {}, {}
            for obj_name in state['objects'].keys():
                obj_state = state['objects'][obj_name]
                pos0 = _to_tensor_pos(obj_state['pos'])
                # 注意：你的 init_state 里姿态字段是 'rot'，后续写回统一为 'quat'
                quat0 = _to_tensor_quat(obj_state.get('quat', obj_state.get('rot')))

                delta_pos = torch.zeros(3)
                delta_q = np.array([0, 0, 0, 1], dtype=np.float32)

                if obj_name in rnd_cfg:
                    if 'pos_randomization' in rnd_cfg[obj_name]:
                        pos_cfg = rnd_cfg[obj_name]['pos_randomization']
                        if pos_cfg['method'] == 'uniform':
                            delta_pos = _uniform_masked(pos_cfg['std'], pos_cfg['along_xyz'])
                        elif pos_cfg['method'] == 'grid':
                            # 新增：网格采样
                            std_val = float(pos_cfg['std'])
                            along = pos_cfg['along_xyz']
                            n_opt = pos_cfg.get('n', None)
                            delta_pos = _grid_masked(std=std_val, along_xyz=along, n=n_opt, max_spacing=0.05)

                        else:
                            raise ValueError(f"unknown pos_randomization method: {pos_cfg['method']}")

                    if 'quat_randomization' in rnd_cfg[obj_name]:
                        quat_cfg = rnd_cfg[obj_name]['quat_randomization']
                        if quat_cfg['method'] == 'uniform':
                            deg_std = float(quat_cfg['std'])
                            axes = quat_cfg['along_xyz']
                            ang = np.zeros(3, dtype=np.float32)
                            for i in range(3):
                                if axes[i]:
                                    ang[i] = np.random.uniform(-deg_std, deg_std) * np.pi / 180.0
                            delta_q = euler_to_quaternion_xyz(ang[0], ang[1], ang[2])
                        else:
                            raise ValueError(f"unknown quat_randomization method: {quat_cfg['method']}")

                new_pos = pos0 + delta_pos
                new_q = _quat_right_mul(quat0.detach().cpu().numpy() if torch.is_tensor(quat0) else quat0, delta_q)
                new_quat = torch.tensor(new_q, dtype=torch.float32)

                updated_objs[obj_name] = (new_pos, new_quat)
                deltas[obj_name] = (delta_pos, delta_q)
            return updated_objs, deltas

        def _pairwise_min_dist_ok(updated_objs, thresh):
            names = list(updated_objs.keys())
            centers = [updated_objs[n][0] for n in names]
            K = len(centers)
            if K <= 1:
                return True
            centers = [c.detach().cpu().float().numpy() if torch.is_tensor(c) else np.asarray(c, dtype=np.float32)
                       for c in centers]
            for i in range(K):
                for j in range(i + 1, K):
                    if np.linalg.norm(centers[i] - centers[j]) < thresh:
                        return False
            return True

        for idx, state in enumerate(init_states):
            # 拒绝采样直到满足 min_sep
            for _ in range(max_retries):
                updated_objs, deltas = _sample_once_and_apply(state, randomization_cfg)
                if _pairwise_min_dist_ok(updated_objs, min_sep):
                    dpos_butter = torch.zeros(3); dq_butter = np.array([0, 0, 0, 1], dtype=np.float32)
                    dpos_basket = torch.zeros(3); dq_basket = np.array([0, 0, 0, 1], dtype=np.float32)

                    for name, (pos_t, quat_t) in updated_objs.items():
                        state['objects'][name]['pos'] = pos_t + 0.005
                        state['objects'][name]['rot'] = quat_t
                        if name in deltas:
                            dp, dq = deltas[name]
                            if name == "butter":
                                dpos_butter, dq_butter = dp, dq
                            elif name == "basket":
                                dpos_basket, dq_basket = dp, dq

                    delta_pos_ee_butter.append(dpos_butter)
                    delta_rot_quat_butter.append(dq_butter)
                    delta_pos_ee_basket.append(dpos_basket)
                    delta_rot_quat_basket.append(dq_basket)
                    break

        init_states, all_actions, all_states, delta_pos_ee_butter, delta_rot_quat_butter, delta_pos_ee_basket, delta_rot_quat_basket, split_idxs_list = repeat_on_rand_states(init_states, all_actions, all_states, delta_pos_ee_butter, delta_rot_quat_butter, delta_pos_ee_basket, delta_rot_quat_basket, split_idxs_list, num_trajs_per_state)
        all_actions_backup = copy.deepcopy(all_actions)
        plot_object_position_distributions(init_states=init_states, save_path=plot_dir)
        if not rand_traj:
            return init_states, None, None
        # 3) IK 求解器
        ik_solver, tensor_args = init_ik_solver(urdf_file, base_link, ee_link, use_cuda_graph=False)

        # 4) 应用到轨迹：仅保留 IK 成功的 demo
        kept_init_states = []
        kept_updated_actions = []
        kept_all_states = []

        for traj_idx, spans in tqdm(enumerate(split_idxs_list), total=len(split_idxs_list)):
            if not spans:
                # 没有非抓/放段，跳过
                continue

            # 更稳健的两段选取：优先 [0] 和 [2]，否则仅 [0]
            use_spans = [spans[0]] + ([spans[2]] if len(spans) >= 3 else [])

            traj = copy.deepcopy(all_actions_backup[traj_idx])
            success = True
            for seg_i, (s, e) in enumerate(use_spans):
                if seg_i == 0:
                    dpos = delta_pos_ee_butter[traj_idx].clone()
                    dq = delta_rot_quat_butter[traj_idx]
                    dpos[-1] = dpos[-1] + 0.005
                    dpos[0] = dpos[0] + 0.02
                    end_gripper_idx = use_spans[seg_i+1][0] if seg_i + 1 < len(use_spans) else len(traj) - 1
                else:
                    dpos = delta_pos_ee_basket[traj_idx].clone()
                    dq = delta_rot_quat_basket[traj_idx]
                    # 这里你原来是“保持不变”，我照抄
                    dpos[0] = dpos[0] + 0.03
                    dpos[1] = dpos[1] + 0.05
                    end_gripper_idx = len(traj) - 1

                new_traj = update_traj(
                    traj,
                    delta_pos_ee=dpos,
                    delta_rot_ee=dq,
                    start_idx=s,
                    end_idx=e,
                    end_gripper_idx=end_gripper_idx,
                    ik_solver=ik_solver,
                    device=tensor_args.device,
                )
                count = 1
                max_count = 3
                while new_traj is None:
                    new_traj = update_traj(
                        traj,
                        delta_pos_ee=dpos,
                        delta_rot_ee=dq,
                        start_idx=s,
                        end_idx=e,
                        end_gripper_idx=end_gripper_idx,
                        ik_solver=ik_solver,
                        device=tensor_args.device,
                    )
                    count += 1
                    if count > max_count:
                        break

                if new_traj is None:
                    print(f"[skip] IK failed on traj {traj_idx}, seg {seg_i}. Skip this demo.")
                    success = False
                    break
                traj = new_traj

            if not success:
                # 这个 demo 直接跳过：不加入 kept_* 列表
                continue
            second_span_start = use_spans[1][0] + 5
            lift_ok = _apply_post_grasp_lift(
                traj,
                second_span_start,
                ik_solver=ik_solver,
                device=tensor_args.device,
                lift=0.20,          # 抬高 0.20 m
                smooth_frames=30,   # 平滑 30 帧，终点为 t+30
                joint_names=EXPECTED_JOINT_NAMES,
            )
            count = 0
            max_count = 5
            while lift_ok is False and count < max_count:
                lift_ok = _apply_post_grasp_lift(
                    traj,
                    second_span_start,
                    ik_solver=ik_solver,
                    device=tensor_args.device,
                    lift=0.20,          # 抬高 0.20 m
                    smooth_frames=30,   # 平滑 30 帧，终点为 t+30
                    joint_names=EXPECTED_JOINT_NAMES,
                )
                count += 1
            if not lift_ok:
                print(f"[skip] Lifting IK failed on traj {traj_idx}. Skip this demo.")
                continue

            kept_updated_actions.append(traj)
            kept_init_states.append(init_states[traj_idx])
            kept_all_states.append(all_states[traj_idx])

        # 5) 保存（仅保存成功样本）
        os.makedirs(save_init_states_dir, exist_ok=True)
        init_states_path = os.path.join(save_init_states_dir, f"{task_name}_init_states.metadata")
        save_init_states(kept_init_states, init_states_path)

        return kept_init_states, kept_updated_actions, kept_all_states

    except Exception as e:
        print(f"Error in randomizing objects and traj: {e}")
        _, _, tb = sys.exc_info()
        pdb.post_mortem(tb)
