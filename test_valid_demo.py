import os
dir = r"roboverse_demo/demo_isaaclab"

dirs = os.listdir(dir)
dirs.sort()
for folder_name in dirs:
    folder = os.path.join(dir, folder_name, "robot-franka")
    print(f"{len(os.listdir(folder))} demos: {folder_name}")
    for demo_idx, demo_folder in enumerate(os.listdir(folder)):
        demo_folder = os.path.join(folder, demo_folder)
        flag = 0
        for file in os.listdir(demo_folder):
            if file.endswith(".json"):
                flag = 1
                break
        if not flag:
            print(f"\t{folder_name}:{demo_idx:04d} has no json file")
