import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.realpath('.'))

import argparse
import multiprocessing
from rich.progress import track

def transfer_demo(demo_dir):
    command = ' '.join([
        'python /home/ghr/yktang/RoboVerse/process_single_libero.py',
        f'--demo_dir {demo_dir}',
    ])
    os.system(command)

if __name__ == '__main__':
    num_proc = 40
    demo_root_dir = "/scratch/current/ghr/szang/szang-workspace/SyntheticVLA-mini/experiments/robot/libero/datasets/libero_90_no_noops_dp/KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet"
    demo_dir_list = sorted(os.listdir(demo_root_dir))
    print(f"Transferring on demos: {demo_dir_list}")
    demo_dir_list = [os.path.join(demo_root_dir, d) for d in demo_dir_list if d.startswith("demo_")]
    # compose scenes in parallel
    with multiprocessing.Pool(num_proc) as pool:
        it = track(
            pool.imap_unordered(transfer_demo, demo_dir_list),
            total=len(demo_dir_list),
            description='Transferring demos...',
        )
        list(it)
