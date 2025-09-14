import os


root_dir = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_libero/ExtLibero100/robot-franka"
subdirs = sorted(os.listdir(root_dir))
for subdir in subdirs:
    if subdir.startswith("demo_"):
        demo_root = os.path.join(root_dir, subdir)
        pcd_path = os.path.join(demo_root, "point_clouds.npy")
        if os.path.exists(pcd_path):
            new_pcd_path = os.path.join(demo_root, "pointclouds.npy")
            os.rename(pcd_path, new_pcd_path)
            print(f"Renamed {pcd_path} to {new_pcd_path}")
