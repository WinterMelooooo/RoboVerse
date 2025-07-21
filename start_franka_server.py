from roboverse_learn.algorithms.utils.franka_ros import FrankaRobotServer
import rclpy

def main():
    rclpy.init()
    server = FrankaRobotServer(socket_number=5555)
    server.run()


if __name__ == "__main__":
    main()
