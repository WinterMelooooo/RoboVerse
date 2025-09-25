import os
import json
base_dir = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_isaaclab/LiberoPickButter-Level4/robot-franka"
merge_dir = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_isaaclab/LiberoPickButter-Level4-save/robot-franka"

def main():
    base_dir_files = os.listdir(base_dir)
    base_dir_files = sorted(base_dir_files)
    base_dir_files = [f for f in base_dir_files if f.startswith("demo_")]
    merge_dir_files = os.listdir(merge_dir)
    merge_dir_files = sorted(merge_dir_files)
    merge_dir_files = [f for f in merge_dir_files if f.startswith("demo_")]


    base_dir_max_num = base_dir_files[-1].split("_")[-1]
    print(f"base_dir_max_num: {base_dir_max_num}")
    for file in merge_dir_files:
        merge_dir_num = file.split("_")[-1]
        merge_dir_num = int(merge_dir_num) + int(base_dir_max_num) + 1
        merge_dir_num = f"{merge_dir_num:04d}"
        print(f"Renaming {file} to {file.replace(file.split('_')[-1], str(merge_dir_num))}")
        os.rename(os.path.join(merge_dir, file), os.path.join(base_dir, file.replace(file.split("_")[-1], str(merge_dir_num))))


if __name__ == "__main__":
    main()
