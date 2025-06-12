from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import (
    get_robot,
    get_sim_env_class,
    get_task,
    get_wrapper_class,
)

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
