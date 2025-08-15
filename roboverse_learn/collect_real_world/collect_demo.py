import sys
sys.path.append('.')
import select
import tty
import termios
import rclpy
import os
import time
from tqdm import tqdm
from std_srvs.srv import SetBool
from roboverse_learn.collect_real_world.state_reader import FrankaStateReader
from roboverse_learn.collect_real_world.demo_saver import save_single_demo
from roboverse_learn.algorithms.utils.multi_realsense import MultiRealsenseWrapper
Taskname = "RealworldLiberoPickButter"


def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def get_keyboard_input():
    if isData():
        c = sys.stdin.read(1)
        return c
    else:
        return None

def set_gello_live(node, cli, live: bool):
    req = SetBool.Request()
    req.data = live
    future = cli.call_async(req)

    rclpy.spin_until_future_complete(node, future)
    print(future.result().success, future.result().message)

def wait_for_first_messages(reader, timeout=5.0):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        rclpy.spin_once(reader, timeout_sec=0.1)
        js, gw = reader.get_state()
        tj, tgw = reader.get_target_state()
        if js is not None and gw is not None and tj is not None and tgw is not None:
            return True
    return False

def main():
    old_settings = termios.tcgetattr(sys.stdin)
    camera = MultiRealsenseWrapper(set_auto_exposure=False, exposure_time=1000, gain = 16)
    rclpy.init()
    state_reader = FrankaStateReader()
    # gello_fix_node = rclpy.create_node('client')
    # cli = gello_fix_node.create_client(SetBool, 'gello/set_live')
    # if not cli.wait_for_service(timeout_sec=5.0):
    #     print("Service not available, exiting...")
    #     rclpy.shutdown()
    #     return
    save_dir = os.path.join("roboverse_demo", "demo-realworld", Taskname, "robot-franka")
    demo_idx = 0
    cache = []
    print("5 seconds warming up for exposure...")
    hz = 30
    sec = 5
    # set_gello_live(gello_fix_node, cli, live=False)
    for i in tqdm(range(hz * sec)):
        vis, _ = camera.read_cameras()
        time.sleep(1.0 / hz)
    ok = wait_for_first_messages(state_reader, timeout=5.0)
    if not ok:
        print("Failed to receive first messages from FrankaStateReader, exiting...")
        rclpy.shutdown()
        return
    print("Press Enter to continue collecting data. Press 'r' to abandon. Press 'q' to quit.")
    tty.setcbreak(sys.stdin.fileno())
    while True:
        key = get_keyboard_input()
        if key == '\n':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            # set_gello_live(gello_fix_node, cli, live=True)
            print("Starting data collection...")
            print("Press 'Enter' to save the current demo, 'r' to abandon this demo and start a new one, or 'q' to quit.")
            break
        elif key == 'q':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("Exiting without collecting data.")
            rclpy.shutdown()
            return
    while True:
        rclpy.spin_once(state_reader, timeout_sec=0.1)
        joint_pos, gripper_width = state_reader.get_state()
        joint_pos_target, gripper_width_target = state_reader.get_target_state()
        vis, _ = camera.read_cameras()

        gripper_pos = [gripper_width/2] * 2
        state = gripper_pos + joint_pos
        gripper_target_pos = [gripper_width_target/2] * 2
        state_target = gripper_target_pos + joint_pos_target
        obs = {
            "robots":{
                "franka":{
                    "dof_pos": state,
                    "dof_pos_target": state_target,
                }
            },
            "cameras": {
                "camera0": {
                    "rgb": vis["camera0"]["rgb"],
                    "depth": vis["camera0"]["depth"],
                    "intrinsics": vis["camera0"]["intrinsics"],
                    "extrinsics": vis["camera0"]["extrinsics"],
                }
            }
        }
        cache.append(obs)
        key = get_keyboard_input()
        if key is not None:
            if key == 'q':
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                print("Exiting...")
                break
            elif key == '\n':
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                # set_gello_live(gello_fix_node, cli, live=False)
                print("Saving collected data...")
                demo_dir = os.path.join(save_dir, f"demo_{demo_idx:04d}")
                os.makedirs(demo_dir, exist_ok=True)
                save_single_demo(demo_dir, cache)
                cache = []
                print(f"\tDemo {demo_idx:04d} saved!")
                print("Press Enter to continue collecting data. Press 'q' to quit.")
                demo_idx += 1
                tty.setcbreak(sys.stdin.fileno())
                while True:
                    key = get_keyboard_input()
                    if key == '\n':
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        # set_gello_live(gello_fix_node, cli, live=True)
                        print(f"Start collecting demo {demo_idx:04d}...")
                        print("Press 'r' to abandon this demo and start a new one. Press 'q' to quit.")
                        tty.setcbreak(sys.stdin.fileno())
                        break
                    elif key == 'q':
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        print("Exiting without collecting more data.")
                        rclpy.shutdown()
                        return
            elif key == 'r':
                cache = []
                # set_gello_live(gello_fix_node, cli, live=False)
                print("Press Enter to start next demo. Press 'q' to quit.")
                while True:
                    key = get_keyboard_input()
                    if key == '\n':
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        # set_gello_live(gello_fix_node, cli, live=True)
                        print(f"Start collecting demo {demo_idx:04d}...")
                        print("Press 'Enter' to save the current demo, 'r' to abandon this demo and start a new one, or 'q' to quit.")
                        tty.setcbreak(sys.stdin.fileno())
                        break
                    elif key == 'q':
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        print("Exiting without collecting more data.")
                        rclpy.shutdown()
                        return

    rclpy.shutdown()

if __name__ == '__main__':
    main()
