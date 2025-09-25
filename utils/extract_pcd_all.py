import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.realpath('.'))

import argparse
import multiprocessing
from rich.progress import track
gpu_list = [0,1,2,3,4,5,6,7]
def transfer_demo(demo_dir):
    # Get worker id
    worker_id = multiprocessing.current_process()._identity[0]
    gpu_id = gpu_list[worker_id % len(gpu_list)]

    command = ' '.join([
        "CUDA_VISIBLE_DEVICES="+str(gpu_id),
        'python /home/ghr/yktang/RoboVerse/extract_pcd_single.py',
        f'--demo_root_dir {demo_dir}',
    ])
    os.system(command)

if __name__ == '__main__':
    demo_root_dir = "/home/ghr/yktang/RoboVerse/roboverse_demo/demo_libero/ExtLibero100/robot-franka"
    demo_dir_list = sorted(os.listdir(demo_root_dir))
    demo_dir_list = [d for d in demo_dir_list if d.startswith("demo_")]
    print(f"Transferring on demos: {demo_dir_list}")
    demo_dir_list = [os.path.join(demo_root_dir, d) for d in demo_dir_list if d.startswith("demo_")]
    # compose scenes in parallel
    with multiprocessing.Pool(len(gpu_list)) as pool:
        it = track(
            pool.imap_unordered(transfer_demo, demo_dir_list),
            total=len(demo_dir_list),
            description='Transferring demos...',
        )
        list(it)
