import json
import os
from collections import defaultdict

from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import (
    get_robot,
    get_sim_env_class,
    get_task,
    get_wrapper_class,
)


def match_indices_by_hash(
    demo_dir,
    init_states,
    robot_name,
    tol=1e-6,  # 坐标容差
):
    # 1) 构建 init_states 位置哈希表
    pos_map = defaultdict(list)  # key: rounded pos tuple -> [idx2,...]
    key_name = f"robot.{robot_name.lower()}.pos"
    for idx2, st in enumerate(init_states):
        # 取出 pos，可以是 list 或 torch.Tensor
        p = st["robots"][robot_name.lower()]["pos"]
        pl = p.tolist() if hasattr(p, "tolist") else list(p)
        key = tuple(x for x in pl)
        pos_map[key].append((idx2, pl))

    # 2) 对每个 demo_dir[idx1] 查表并做容差筛选
    result = {}
    for idx1, demo_folder in enumerate(demo_dir):
        # demo.robot_root_state[0][:3] 可能是 list/tuple/np.ndarray
        metadata_path = f"{demo_folder}/metadata.json"
        demo = json.load(open(metadata_path, "r"))
        p0 = demo["robot_root_state"][0][:3]
        pl0 = p0.tolist() if hasattr(p0, "tolist") else list(p0)
        key0 = tuple(x for x in pl0)

        matches = []
        for idx2, pl in pos_map.get(key0, []):
            # 逐坐标比较误差
            if all(abs(a - b) <= tol for a, b in zip(pl0, pl)):
                matches.append(idx2)

        result[idx1] = matches

    return result


if __name__ == "__main__":
    task_name = "LiberoPickAlphabetSoup"
    robot_name = "Franka"
    demo_dirs = [
        r"roboverse_demo/demo_isaaclab/LiberoPickAlphabetSoup-Level0",
        r"roboverse_demo/demo_isaaclab/LiberoPickAlphabetSoup-Level1",
        r"roboverse_demo/demo_isaaclab/LiberoPickAlphabetSoup-Level2",
    ]
    for idx, demo_dir in enumerate(demo_dirs):
        demo_dirs[idx] += f"/robot-{robot_name.lower()}"

    env = None
    task = get_task(task_name)
    robot = get_robot(robot_name)

    init_states, all_actions, all_states = get_traj(task, robot, env)

    # if match:
    # demo_dir[idx1].robot_root_state[0][:3] == init_states[idx2]["robot"]["f{robot_name.lower()}"]["pos"]
    for demo_dir in demo_dirs:
        print(f"Processing {demo_dir}...")
        sub_demo_dirs = os.listdir(demo_dir)
        sub_demo_dirs = [os.path.join(demo_dir, d) for d in sub_demo_dirs if os.path.isdir(os.path.join(demo_dir, d))]
        match = match_indices_by_hash(
            sub_demo_dirs,
            init_states,
            robot_name,
            tol=1e-8,  # 坐标容差
        )
        print(match)
