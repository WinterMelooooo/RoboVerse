import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.realpath('.'))

import argparse
import multiprocessing
from rich.progress import track


demo_root = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_realworld/RealworldPickBottle/robot-franka"
dest_root = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_polymetis"
task_name = demo_root.split("/")[-2]
dest_root = os.path.join(dest_root, task_name)
os.makedirs(dest_root, exist_ok=True)


def transfer_demo(demo_idx):
    demo_dir = os.path.join(demo_root, f"demo_{demo_idx:04d}")
    dest_dir = os.path.join(dest_root, f"demo_{demo_idx:04d}")
    command = ' '.join([
        'python /home/ghr/yktang/RoboVerse/roboverse_to_polymetis.py',
        f'--demo_dir {demo_dir}',
        f'--dest_dir {dest_dir}',
    ])
    os.system(command)

if __name__ == '__main__':
    num_proc = 50
    demo_id_list = os.listdir(demo_root)
    demo_id_list = [int(demo_id.split("demo_")[-1]) for demo_id in demo_id_list if demo_id.startswith("demo_")]
    print(f"Transferring on demos: {demo_id_list}")
    # compose scenes in parallel
    with multiprocessing.Pool(num_proc) as pool:
        it = track(
            pool.imap_unordered(transfer_demo, demo_id_list),
            total=len(demo_id_list),
            description='Transferring demos...',
        )
        list(it)
