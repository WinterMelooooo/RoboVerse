from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task

robot_name = "franka"
robot = get_robot(robot_name)
task_name = "CloseBox"
task = get_task(task_name)


init_states, all_actions, all_states = get_traj(task, robot)
print(f"len(init_states) = {len(init_states)}")
