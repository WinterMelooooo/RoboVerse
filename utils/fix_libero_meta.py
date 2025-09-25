import json
import os
import shutil


# src_dir = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_libero/ExtLibero100/robot-franka"
# root_dir = src_dir.split("/ExtLibero100")[0]
# split_demo_idx = 47 + 1
# names = ["KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet", "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet"]

# folders = sorted(os.listdir(src_dir))
# # prev = folders[:split_demo_idx]
# # after = folders[split_demo_idx:]

# # for folder in prev:
# #     task_name = names[0]
# #     meta_path = os.path.join(src_dir, folder, "metadata.json")
# #     data = json.load(open(meta_path, "r"))
# #     action = data["action"]
# #     tgt = data["joint_qpos_target"]
# #     data["joint_qpos_target"] = action
# #     with open(meta_path, "w") as f:
# #         json.dump(data, f, indent=4)
# #     print(f"Fixed {meta_path}, set joint_qpos_target to action")

# #     dest_folder_path = os.path.join(root_dir, task_name, folder)
# #     shutil.move(os.path.join(src_dir, folder), dest_folder_path)


# for folder in folders:
#     task_name = names[1]
#     meta_path = os.path.join(src_dir, folder, "metadata.json")
#     data = json.load(open(meta_path, "r"))
#     action = data["action"]
#     tgt = data["joint_qpos_target"]
#     data["joint_qpos_target"] = action
#     with open(meta_path, "w") as f:
#         json.dump(data, f, indent=4)
#     print(f"Fixed {meta_path}, set joint_qpos_target to action")

#     dest_folder_path = os.path.join(root_dir, task_name, folder)
#     shutil.move(os.path.join(src_dir, folder), dest_folder_path)


src = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_libero/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet"
folders = sorted(os.listdir(src))
for folder in folders:
    idx = int(folder.split("_")[-1])
    new_idx = idx-48

    new_folder_name = f"demo_{new_idx:04d}"
    shutil.move(os.path.join(src, folder), os.path.join(src, new_folder_name))
    print(f"Renamed {folder} to {new_folder_name}")
