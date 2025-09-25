import os
import imageio as iio

root_dir = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_realworld"
save_dir = "/home/ghr/yktang/RoboVerse/tmp/demo"
os.makedirs(save_dir, exist_ok=True)
stride = 5


task_names = sorted(os.listdir(root_dir))
root_dirs = [os.path.join(root_dir, task_name) for task_name in task_names]
save_dirs = [os.path.join(save_dir, task_name) for task_name in task_names]
for idx, root_dir in enumerate(root_dirs):
    save_dir = save_dirs[idx]
    os.makedirs(save_dir, exist_ok=True)
    root_dir = os.path.join(root_dir, "robot-franka", "demo_0000")
    rgb_file = os.path.join(root_dir, "rgb.mp4")
    rgbs = iio.mimread(rgb_file, memtest=False)
    for i in range(0, len(rgbs), stride):
        save_path = os.path.join(save_dir, f"{i:04d}.png")
        iio.imwrite(save_path, rgbs[i])
    print(f"Saved {len(rgbs)//stride} images to {save_dir}")
