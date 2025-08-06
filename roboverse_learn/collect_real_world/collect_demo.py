import sys
sys.path.append('.')
import select
import tty
import termios
import rclpy
import os
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


def main():
    old_settings = termios.tcgetattr(sys.stdin)
    camera = MultiRealsenseWrapper()
    rclpy.init()
    state_reader = FrankaStateReader()
    save_dir = os.path.join("roboverse_demo", "demo-realworld", Taskname, "robot-franka")
    demo_idx = 0
    cache = []
    print("Press Enter to continue collecting data. Press 'r' to abandon. Press 'q' to quit.")
    tty.setcbreak(sys.stdin.fileno())
    while True:
        key = get_keyboard_input()
        if key == '\n':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("Starting data collection...")
            break
        elif key == 'q':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("Exiting without collecting data.")
            rclpy.shutdown()
            return
    while True:
        rclpy.spin_once(state_reader, timeout_sec=0.1)
        joint_pos, gripper_width = state_reader.get_state()
        vis, _ = camera.read_cameras()

        gripper_pos = [gripper_width/2] * 2
        state = gripper_pos + joint_pos
        print(state)
        obs = {
            "robots":{
                "franka":{
                    "dof_pos": state,
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
                print("Saving collected data...")
                demo_dir = os.path.join(save_dir, f"demo_{demo_idx:04d}")
                os.makedirs(demo_dir, exist_ok=True)
                save_single_demo(demo_dir, cache)
                demo_idx += 1
                cache = []
                print(f"\tDemo {demo_idx:04d} saved!")
                print("Press Enter to continue collecting data. Press 'r' to abandon. Press 'q' to quit.")
                tty.setcbreak(sys.stdin.fileno())
                while True:
                    key = get_keyboard_input()
                    if key == '\n':
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        print(f"Start collecting demo {demo_idx:04d}...")
                        tty.setcbreak(sys.stdin.fileno())
                        break
                    elif key == 'q':
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        print("Exiting without collecting more data.")
                        rclpy.shutdown()
                        return
            elif key == 'r':
                cache = []
                print("Press Enter to start next demo. Press 'q' to quit.")
                while True:
                    key = get_keyboard_input()
                    if key == '\n':
                        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                        print(f"Start collecting demo {demo_idx:04d}...")
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
