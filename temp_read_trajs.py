from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import get_robot, get_sim_env_class, get_task
robot_name = "franka"
robot = get_robot(robot_name)
env = None
root_dir = r"roboverse_data/trajs"
