import json
import os

import imageio as iio

from metasim.cfg.randomization import RandomizationCfg
from metasim.cfg.render import RenderCfg
from metasim.cfg.robots.base_robot_cfg import BaseRobotCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import SimType
from metasim.sim import BaseSimHandler, EnvWrapper
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task
from metasim.utils.state import state_tensor_to_nested

metadata_path = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_realworld/RealworldLiberoPickButter/robot-franka/demo_0000/metadata.json"

sim = "isaaclab"
robot = "franka"
task = "LiberoPickButter"
num_env = 1

handler_class = get_sim_env_class(SimType(sim))
task = get_task(task)
robot = get_robot(robot)
camera = PinholeCameraCfg(data_types=["rgb", "depth"], pos=(1.5, 0.0, 1.5), look_at=(0.0, 0.0, 0.0))
scenario = ScenarioCfg(
    task=task,
    robots=[robot],
    scene=None,
    cameras=[camera],
    random=RandomizationCfg(),
    try_add_table=True,
    render=RenderCfg(mode="raytracing"),
    split="all",
    sim=sim,
    headless=True,
    num_envs=num_env,
)
env = handler_class(scenario)

data = json.load(open(metadata_path, "r"))
actions = data["joint_qpos_target"]
# data_input_seq = [
#     "panda_finger_joint1",
#     "panda_finger_joint2",
#     "panda_joint1",
#     "panda_joint3",
#     "panda_joint6",
#     "panda_joint7",
#     "panda_joint2",
#     "panda_joint4",
#     "panda_joint5",
# ]
data_input_seq = [
    "panda_finger_joint1",
    "panda_finger_joint2",
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
demo_idxs = [0]  # You can change this to any demo index you want to test
init_states, all_actions, all_states = get_traj(task, robot, env.handler)

tot_demo = len(all_actions)

init_states = init_states[: int(tot_demo * 0.9)]
all_actions = all_actions[: int(tot_demo * 0.9)]
all_states = all_states[: int(tot_demo * 0.9)]


obs, extras = env.reset(states=[init_states[demo_idx] for demo_idx in demo_idxs])
## Initialize
rgbs = []
for action in actions:
    action = {k: v for k, v in zip(data_input_seq, action)}
    action = [{"franka": {"dof_pos_target": action}}]
    obs, reward, success, time_out, extras = env.step(action)
    obs = state_tensor_to_nested(env.handler, obs)
    rgb = obs[0]["cameras"]["camera0"]["rgb"]
    rgbs.append(rgb.cpu().numpy())

iio.mimsave(os.path.join("/home/ghr/yktang/RoboVerse/tmp", "replay.mp4"), rgbs, fps=30, quality=10)
env.close()
