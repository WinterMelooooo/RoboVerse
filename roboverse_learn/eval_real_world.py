from __future__ import annotations

import datetime
import os
import time
from dataclasses import dataclass
from typing import Literal

from roboverse_learn.algorithms.utils.real_world_env import RealWorldEnv

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass

import select

import imageio.v2 as iio
import numpy as np
import rootutils
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])

from PIL import Image
from termcolor import cprint

from metasim.cfg.randomization import RandomizationCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors.cameras import PinholeCameraCfg
from metasim.constants import SimType
from metasim.utils.demo_util import get_traj
from metasim.utils.setup_util import (
    get_robot,
    get_sim_env_class,
    get_task,
    get_wrapper_class,
)
from roboverse_learn.algorithms import PolicyRunner, get_runner
from roboverse_learn.algorithms.utils.train_data_selector import reorder_init_states


@dataclass
class Args:
    random: RandomizationCfg
    """Domain randomization options"""
    task: str
    """Task name"""
    robot: str = "franka"
    """Robot name"""
    num_envs: int = 1
    """Number of parallel environments, find a proper number for best performance on your machine"""
    sim: Literal["isaaclab", "mujoco", "isaacgym"] = "isaaclab"
    """Simulator backend"""
    max_demo: int | None = None
    """Maximum number of demos to collect, None for all demos"""
    headless: bool = False
    """Run in headless mode"""
    table: bool = True
    """Try to add a table"""
    task_id_range_low: int = 0
    """Low end of the task id range"""
    task_id_range_high: int = 1000
    """High end of the task id range"""
    checkpoint_path: str = ""
    """Path to the checkpoint"""
    algo: str = "diffusion_policy"
    """Algorithm to use"""
    subset: str = "pickcube_l0"
    """Subset your ckpt trained on"""
    action_set_steps: int = 1
    """Number of steps to take for each action set"""
    save_video_freq: int = 1
    """Frequency of saving videos"""
    max_step: int = 1000
    """Maximum number of steps to collect"""
    gpu_id: int = 0
    """GPU ID to use"""
    wrapper_class: str | None = None
    """Env wrapper to use"""
    use_touch: bool = False
    """Use touch sensor"""

    def __post_init__(self):
        if self.random.table and not self.table:
            log.warning("Cannot enable table randomization without a table, disabling table randomization")
            self.random.table = False
        log.info(f"Args: {self}")


args = tyro.cli(Args)

DEBUG_RGB = False
DEBUG_RAND_STATE = False
DEBUG_PCD = False


def main():
    num_envs: int = args.num_envs
    log.info(f"Using GPU device: {args.gpu_id}")

    task = get_task(args.task)
    task.episode_length = args.action_set_steps * args.max_step
    robot = get_robot(args.robot)
    env = RealWorldEnv()
    task.episode_length = args.action_set_steps * args.max_step

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ckpt_name = args.checkpoint_path.split("/")[-1] + "_" + time_str
    ckpt_name = f"{args.task}/{args.algo}/{args.robot}/L{args.random.level}/{ckpt_name}"
    runnerCls = get_runner(args.algo)

    # THIS IS ONLY FOR ALIGNING WITH INTERFACE, DO NOT USE IN REAL WORLD EVAL
    camera = PinholeCameraCfg(pos=(1.5, 0, 1.5), look_at=(0.0, 0.0, 0.0))
    scenario = ScenarioCfg(
        task=args.task,
        robots=[args.robot],
        cameras=[camera],
        random=args.random,
        sim=args.sim,
        num_envs=args.num_envs,
        headless=args.headless,
    )
    # WARNING ENDS

    policyRunner: PolicyRunner = runnerCls(
        scenario=scenario,
        num_envs=num_envs,
        checkpoint_path=args.checkpoint_path,
        device=f"cuda:{args.gpu_id}",
        task_name=args.task,
        subset=args.subset,
    )
    action_set_steps = 2 if policyRunner.policy_cfg.action_config.action_type == "ee" else 1
    if use_pcd(policyRunner.yaml_cfg):
        try:
            from roboverse_learn.algorithms.utils.pnt_cloud_getter import PntCloudGetter
        except:
            import sys

            sys.path.append(".")
            from roboverse_learn.algorithms.utils.pnt_cloud_getter import PntCloudGetter
        pnt_cloud_getter = PntCloudGetter(args.task.split("_")[0], use_point_crop=True)
        """
        temp_dict = {
            "cam_pos": [1.5, 0.0, 1.5],
            "cam_look_at": [0.0, 0.0, 0.0],
            "cam_intr": torch.tensor(
                [
                    [293.19970703125, 0.0, 128.0],
                    [0.0, 293.19970703125, 128.0],
                    [0.0, 0.0, 1.0],
                ]
            ).cpu(),
            "cam_extr": torch.tensor(
                [
                    [0.0, 1.0, -0.0, -0.0],
                    [0.7071067690849304, -0.0, -0.7071067690849304, -0.0],
                    [-0.7071068286895752, 0.0, -0.7071068286895752, 2.1213204860687256],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ).cpu(),
        }
        """

    total_success = 0
    total_completed = 0
    if args.max_demo is None:
        max_demos = args.task_id_range_high - args.task_id_range_low
    else:
        max_demos = args.max_demo

    for demo_start_idx in range(args.task_id_range_low, args.task_id_range_low + max_demos, num_envs):
        demo_end_idx = min(demo_start_idx + num_envs, max_demos)
        ## Reset before first step
        tic = time.time()
        obs = env.reset()

        policyRunner.reset()
        toc = time.time()
        log.trace(f"Time to reset: {toc - tic:.2f}s")
        log.info(f"ckpt to eval: {args.checkpoint_path.split('/')[-3]}")
        step = 0
        MaxStep = args.max_step
        SuccessOnce = [False] * num_envs
        TimeOut = [False] * num_envs
        images_list = []
        while step < MaxStep:
            log.debug(f"Step {step}")
            new_obs = {
                "rgb": obs.cameras["camera0"].rgb,
                "joint_qpos": obs.robots[args.robot].joint_pos,
            }
            if use_rgbd(policyRunner.yaml_cfg):
                new_obs["depth"] = obs.cameras["camera0"].depth  # (50, 256, 256, 1)
                assert new_obs["depth"].shape[3] == 1, f"Depth should be 1 channels, but got {new_obs['depth'].shape}"
            if use_pcd(policyRunner.yaml_cfg):
                depth = obs.cameras["camera0"].depth
                cam_intr = obs.cameras["camera0"].intrinsics
                cam_extr = obs.cameras["camera0"].extrinsics
                pnt_cloud = pnt_cloud_getter.get_point_cloud(
                    new_obs["rgb"],
                    depth,
                    cam_intr.cpu(),
                    cam_extr.cpu(),
                )
                if DEBUG_RGB:
                    save_dir = f"./tmp/visualize/{args.task}L{args.random.level}"
                    for i in range(num_envs):
                        # 取第 i 个 env 的 rgb 图像 (shape: (H, W, 3))
                        img = np.array(obs.cameras["camera0"].rgb[i].cpu())
                        depth = np.array(obs.cameras["camera0"].depth[i].cpu())
                        # 为每个 demo 创建子目录
                        demo_idx = demo_start_idx + i
                        demo_dir = save_dir
                        os.makedirs(demo_dir, exist_ok=True)
                        # 保存为 PNG
                        file_path = os.path.join(demo_dir, f"demo_{demo_idx:04d}.png")
                        iio.imwrite(file_path, img)
                        depth_file_path = os.path.join(demo_dir, f"demo_{demo_idx:04d}_depth.png")

                        # 假设 depth 是 numpy 数组，dtype 例如 float32 或 uint16
                        depth_min, depth_max = depth.min(), depth.max()
                        if depth_max > depth_min:
                            depth_norm = (depth - depth_min) / (depth_max - depth_min)
                        else:
                            # 全零或常数图像
                            depth_norm = np.zeros_like(depth)
                        # 归一化到 0–255，再转 uint8
                        depth_uint8 = (depth_norm * 255).astype(np.uint8)
                        depth_uint8 = np.squeeze(depth_uint8)

                        depth_img = Image.fromarray(depth_uint8, mode="L")
                        depth_img.save(depth_file_path)
                    # env.close()
                    # raise NotImplementedError()

                if DEBUG_PCD:
                    save_folder = f"./tmp/visualize/{args.task}L{args.random.level}"
                    os.makedirs(save_folder, exist_ok=True)
                    for idx, single_pcd in enumerate(pnt_cloud):
                        pcd_filename = os.path.join(save_folder, f"demo_{idx:04d}_step_{step}.npy")
                        np.save(pcd_filename, single_pcd)
                    # env.close()
                    # raise NotImplementedError("DEBUG")

                new_obs["point_cloud"] = pnt_cloud
                if not use_spUnet_pcd(policyRunner.yaml_cfg):
                    feat_dim = get_pnt_cloud_feat_dim(policyRunner.yaml_cfg)
                    new_obs["point_cloud"] = new_obs["point_cloud"][..., :feat_dim]

            if use_sensor(policyRunner.yaml_cfg):
                new_obs["sensors"] = obs.sensors  # {sensor_name: {"force": Tensor[N_env, 3]}}

            images_list.append(np.array(new_obs["rgb"].cpu()))
            # for key, value in new_obs.items():
            #    print(f"Key: {key}, Value shape: {value.shape}")
            action = policyRunner.get_action(new_obs)
            for round_i in range(action_set_steps):
                obs = env.step(action)
                print("Press ENTER if success", end="", flush=True)
                ready, _, _ = select.select([sys.stdin], [], [], 0)
                if ready:
                    _ = sys.stdin.readline()  # 读掉那一行
                    success = [True]
                else:
                    success = [False]
                time_out = [step >= MaxStep - 1] * num_envs

            # eval
            SuccessOnce = [SuccessOnce[i] or success[i] for i in range(num_envs)]
            TimeOut = [TimeOut[i] or time_out[i] for i in range(num_envs)]
            step += 1
            if all(SuccessOnce):
                break

        SuccessEnd = success.tolist()
        total_success += SuccessOnce.count(True)
        total_completed += len(SuccessOnce)
        os.makedirs(f"tmp/{ckpt_name}", exist_ok=True)
        for i, demo_idx in enumerate(range(demo_start_idx, demo_end_idx)):
            demo_idx_str = str(demo_idx).zfill(4)
            if i % args.save_video_freq == 0:
                iio.mimwrite(
                    f"tmp/{ckpt_name}/{demo_idx}.mp4",
                    [images[i] for images in images_list],
                )
            with open(f"tmp/{ckpt_name}/{demo_idx_str}.txt", "w") as f:
                f.write(f"Demo Index: {demo_idx}\n")
                f.write(f"Num Envs: {num_envs}\n")
                f.write(f"SuccessOnce: {SuccessOnce[i]}\n")
                f.write(f"SuccessEnd: {SuccessEnd[i]}\n")
                f.write(f"TimeOut: {TimeOut[i]}\n")
                f.write(f"Cumulative Average Success Rate: {total_success / total_completed}\n")
                f.write(f"Evaling checkpoint: {args.checkpoint_path}")
        log.info("Demo Indices: ", range(demo_start_idx, demo_end_idx))
        log.info("Num Envs: ", num_envs)
        log.info(f"SuccessOnce: {SuccessOnce}")
        log.info(f"SuccessEnd: {SuccessEnd}")
        log.info(f"TimeOut: {TimeOut}")
        log.info(f"Finished evaling checkpoint: {'/'.join(args.checkpoint_path.split('/')[-4:-2])}")
        log.info(f"Results saved to tmp/{ckpt_name}")
        input("Waiting for reset environment, press ENTER to start the next demo")

    log.info(f"FINAL RESULTS: {total_success / total_completed}")
    with open(f"tmp/{ckpt_name}/final_stats.txt", "w") as f:
        f.write(f"Total Success: {total_success}\n")
        f.write(f"Total Completed: {total_completed}\n")
        f.write(f"Average Success Rate: {total_success / total_completed}\n")
        f.write(f"ckpt: {args.checkpoint_path}\n")
        f.write(f"random level: {args.random.level}\n")
    env.close()


def use_rgbd(cfg):
    task = cfg.get("task", None)
    if task is not None:
        return "head_cam" in cfg.task.shape_meta.obs.keys() and (
            cfg.task.shape_meta.obs.head_cam.type == "rgbd" or cfg.task.shape_meta.obs.head_cam.type == "rgbd_resnet"
        )
    else:
        keys = cfg.dataset.obs_keys.keys()
        return "head_camera_depth" in keys


def use_pcd(cfg):
    task = cfg.get("task", None)
    if task is not None:
        return "point_cloud" in cfg.task.shape_meta.obs.keys() or "pcds" in cfg.task.shape_meta.obs.keys()
    else:
        keys = cfg.dataset.obs_keys.keys()
        return "point_cloud" in keys or "pcds" in keys or "head_camera_pnt_cloud" in keys


def use_dp3_pcd(cfg):
    task = cfg.get("task", None)
    if task is not None:
        return "pcds" in cfg.task.shape_meta.obs.keys()
    else:
        keys = cfg.dataset.obs_keys.keys()
        return (
            "head_camera_pnt_cloud" in keys
            and not cfg.dataset.obs_keys.head_camera_pnt_cloud.get("type", None) == "spUnet"
        )


def use_spUnet_pcd(cfg):
    task = cfg.get("task", None)
    if task is not None:
        return "point_cloud" in cfg.task.shape_meta.obs.keys()
    else:
        keys = cfg.dataset.obs_keys.keys()
        return (
            "head_camera_pnt_cloud" in keys and cfg.dataset.obs_keys.head_camera_pnt_cloud.get("type", None) == "spUnet"
        )


def use_sensor(cfg):
    task = cfg.get("task", None)
    if task is not None:
        return "franka_panda_leftfinger_touch_sensor_pred" in cfg.task.shape_meta.obs.keys()
    else:
        keys = cfg.dataset.obs_keys.keys()
        return "sensors" in keys


def get_pnt_cloud_feat_dim(cfg):
    task = cfg.get("task", None)
    if task is not None:
        return task.shape_meta.obs.point_cloud.shape[-1]
    else:
        return cfg.dataset.obs_keys.head_camera_pnt_cloud.shape[-1]


if __name__ == "__main__":
    main()
