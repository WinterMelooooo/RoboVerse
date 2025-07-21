import os
import pty
import signal
import shutil
import sys
import time

# Commands to run
ROBOT_CMD = [
    "ros2", "launch",
    "franka_fr3_moveit_config", "moveit.launch.py",
    "robot_ip:=172.16.0.2"
]
GRIPPER_CMD = [
    "ros2", "launch",
    "franka_gripper", "gripper.launch.py",
    "robot_ip:=172.16.0.2",
    "use_fake_hardware:=false",
    "arm_id:=fr3",
    "namespace:=gripper"
]

# Combined LD_LIBRARY_PATH prefix
LD_PREFIX = "/usr/lib/x86_64-linux-gnu:/home/user/miniconda3/lib"

def spawn_with_pty(cmd, label):
    """Fork+execvp into a PTY, printing child output with a label."""
    pid, fd = pty.fork()
    if pid == 0:
        # In child: set up environment, then exec
        os.environ["LD_LIBRARY_PATH"] = LD_PREFIX + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        os.execvp(cmd[0], cmd)
    else:
        # In parent: read from PTY
        try:
            while True:
                data = os.read(fd, 1024)
                if not data:
                    break
                sys.stdout.write(f"[{label}] {data.decode(errors='ignore')}")
                sys.stdout.flush()
        except OSError:
            pass
        finally:
            os.close(fd)
        os._exit(0)

def main():
    # Ensure ros2 is on PATH
    if not shutil.which("ros2"):
        print("❌ Cannot find 'ros2' executable on PATH", file=sys.stderr)
        sys.exit(1)

    print("✅ Starting robot and gripper launches with PTYs. Ctrl+C to stop.\n")

    # Spawn robot launch in a child process
    pid = os.fork()
    if pid == 0:
        spawn_with_pty(ROBOT_CMD, "ROBOT")
    else:
        # In parent, spawn gripper after
        time.sleep(3)  # Give robot some time to start
        try:
            spawn_with_pty(GRIPPER_CMD, "GRIPPER")
        except KeyboardInterrupt:
            print("\n⛔ Ctrl+C pressed. Terminating robot launch...")
            os.kill(pid, signal.SIGTERM)
            sys.exit(0)

if __name__ == "__main__":
    main()
