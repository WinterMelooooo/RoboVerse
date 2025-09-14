import os
import shutil

demo_root_dir = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_libero"
srcs_dirs = ["KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet", "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet"]
dest_dir = "Ext_Libero_100"
os.makedirs(os.path.join(demo_root_dir, dest_dir), exist_ok=True)

idx = 0
for src_dir in srcs_dirs:
    src_path = os.path.join(demo_root_dir, src_dir)
    demo_list = sorted(os.listdir(src_path))
    for demo in demo_list:
        src_demo_path = os.path.join(src_path, demo)
        dest_demo_path = os.path.join(demo_root_dir, dest_dir, f"demo_{idx:04d}")
        shutil.copytree(src_demo_path, dest_demo_path)
        print(f"Copied {src_demo_path} to {dest_demo_path}")
        idx += 1
