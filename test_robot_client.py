from roboverse_learn.algorithms.utils.franka_ros_client import FrankaRobotClient

def main():
    client = FrankaRobotClient(socket_number=5555)
    client.do_homing()

if __name__ == "__main__":
    main()
