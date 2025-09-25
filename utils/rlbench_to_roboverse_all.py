import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.realpath('.'))

import argparse
import multiprocessing
from rich.progress import track

def transfer_demo(demo_idx):
    command = ' '.join([
        'python /home/ghr/yktang/RoboVerse/rlbench_to_roboverse.py',
        f'--demo_idx {demo_idx}',
    ])
    os.system(command)

if __name__ == '__main__':
    num_proc = 40
    demo_id_list = os.listdir("/home/ghr/yktang/RoboVerse/third_party/rlbench-18-tasks/data/train/sweep_to_dustpan_of_size/all_variations/episodes")
    demo_id_list = [int(demo_id.split("episode")[-1]) for demo_id in demo_id_list if demo_id.startswith("episode")]
    print(f"Transferring on demos: {demo_id_list}")
    # compose scenes in parallel
    with multiprocessing.Pool(num_proc) as pool:
        it = track(
            pool.imap_unordered(transfer_demo, demo_id_list),
            total=len(demo_id_list),
            description='Transferring demos...',
        )
        list(it)
