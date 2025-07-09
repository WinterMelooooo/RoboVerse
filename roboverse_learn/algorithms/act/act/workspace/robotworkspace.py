import json
import math
import os
import pathlib
import pickle
from copy import deepcopy

import hydra
import IPython
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml
from act.common.json_logger import JsonLogger
from act.utils import dict_apply
from act.workspace.base_workspace import BaseWorkspace
from einops import rearrange
from omegaconf import OmegaConf
from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.model.common.lr_scheduler import (
    get_composite_scheduler,
)
from tqdm import tqdm

from ..constants import DT, PUPPET_GRIPPER_JOINT_OPEN
from ..policy import ACTPolicy, CNNMLPPolicy
from ..utils import (  # helper functions
    compute_dict_mean,
    detach_dict,
    load_data,  # data functions
    set_seed,
)

e = IPython.embed
from datetime import datetime

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class RobotWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None, local_rank=0, world_size=1):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        # Multi GPU lr
        cfg.policy.params.lr *= math.sqrt(world_size)
        cfg.policy.params.lr_backbone *= math.sqrt(world_size)
        OmegaConf.set_struct(cfg.logging, False)
        self.logger = cfg.logging.pop("logger_name", "wandb")
        OmegaConf.set_struct(cfg.logging, True)
        super().__init__(cfg, output_dir=output_dir)
        seed = cfg.training.seed
        set_seed(seed)
        policy = hydra.utils.instantiate(cfg.policy)
        policy.cuda()
        self.policy = policy

    def run(self):
        cfg = self.cfg

        # Logging setup
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        if self.local_rank == 0:
            self.json_logger = JsonLogger(log_path)
            self.json_logger.start()
        self.wandb_run = None
        if self.local_rank == 0:
            if cfg.logging.mode == "online":
                self.wandb_run = wandb.init(
                    dir=str(self.output_dir),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    **cfg.logging,  # project/name/tags/mode/id/group 等
                )
                wandb.config.update({"output_dir": self.output_dir})
        train_dataloader, val_dataloader, stats, _ = load_data(
            cfg.dataset.dataset_dir,
            cfg.dataset.num_episodes,
            cfg.dataset.camera_names,
            cfg.dataset.batch_size_train,
            cfg.dataset.batch_size_val,
            seed=cfg.training.seed,
            keys=cfg.dataset.obs_keys,
            task_name=cfg.task_name.split("_")[0],
            voxel_pnt_cloud=cfg.dataset.voxel_pnt_cloud,
            transform_pcd=cfg.dataset.transform_pcd,
            pnt_dim=cfg.dataset.pnt_dim,
        )

        # save dataset stats
        stats_path = os.path.join(self.output_dir, f"dataset_stats.pkl")
        with open(stats_path, "wb") as f:
            pickle.dump(stats, f)

        best_ckpt_info = self.train_bc(train_dataloader, val_dataloader)
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")

    def train_bc(self, train_dataloader, val_dataloader):
        local_rank = self.local_rank
        global_step = 0
        cfg = self.cfg
        num_epochs = cfg.training.num_epochs
        policy = self.policy
        optimizer = policy.configure_optimizers()
        self.lr_scheduler = None
        if cfg.training.get("lr_scheduler") is not None:
            self.lr_scheduler = get_composite_scheduler(
                cfg.training.lr_scheduler,
                optimizer=optimizer,
                default_num_warmup_steps=cfg.training.lr_warmup_steps
                // self.world_size,
                default_num_training_steps=(
                    len(train_dataloader) * cfg.training.num_epochs
                ),
                # pytorch assumes stepping LRScheduler every epoch
                # however huggingface diffusers steps it every batch
                last_epoch=global_step - 1,
                step_per_epoch=len(train_dataloader),
                params_groups=cfg.get("param_groups", None),
                fill=True,
            )
        policy = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
        min_val_loss = np.inf
        best_ckpt_info = None
        if local_rank == 0:
            epoch_iter = tqdm(range(num_epochs), desc="Epoch", leave=False)
        else:
            epoch_iter = range(num_epochs)
        for epoch in epoch_iter:
            if local_rank == 0:
                epoch_iter.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            train_dataloader.sampler.set_epoch(epoch)
            val_dataloader.sampler.set_epoch(epoch)

            # training
            policy.train()
            optimizer.zero_grad()
            train_losses = []
            for batch_idx, obs in enumerate(train_dataloader):
                obs = train_dataloader.dataset.post_process(obs)
                forward_dict = self.forward_pass(obs, policy)
                # backward
                loss = forward_dict["loss"]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                step_log = {
                    "train_loss": loss.item(),
                    "global_step": global_step,
                    "epoch": epoch,
                }
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                    all_lrs = self.lr_scheduler.get_last_lr()
                    for pg, lr in zip(optimizer.param_groups, all_lrs):
                        step_log[f"lr_{pg.get('name', 'default')}"] = lr
                train_losses.append(loss.item())
                if self.local_rank == 0:
                    self.json_logger.log(step_log)
                    if self.wandb_run is not None:
                        self.wandb_run.log(step_log, step=global_step)
                global_step += 1

            # validation
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for batch_idx, obs in enumerate(val_dataloader):
                    obs = val_dataloader.dataset.post_process(obs)
                    forward_dict = self.forward_pass(obs, policy)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                epoch_val_loss = epoch_summary["loss"]

                step_log["val_loss"] = epoch_val_loss
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (
                        epoch,
                        min_val_loss,
                        deepcopy(policy.state_dict()),
                    )
                    if self.local_rank == 0:
                        best_path = self.save_checkpoint(
                            os.path.join(
                                cfg.checkpoint.save_root_dir, "checkpoints/best.ckpt"
                            )
                        )
                        epoch_iter.write(
                            f"[Epoch {epoch}] New best val loss {min_val_loss:.4f}, saved to {best_path}"
                        )

            # End of epoch
            if local_rank == 0:
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss
                epoch_iter.set_postfix(loss=train_loss.item(), refresh=False)
                self.json_logger.log(step_log)
                if self.wandb_run is not None:
                    self.wandb_run.log(step_log, step=global_step)

            # checkpoint
            if (
                ((epoch + 1) % cfg.training.checkpoint_every) == 0
                or (epoch + 1) == cfg.training.num_epochs
            ) and local_rank == 0:
                self.save_checkpoint(
                    cfg.checkpoint.save_root_dir + f"/checkpoints/{epoch + 1}.ckpt"
                )

        if self.local_rank == 0:
            # json_logger.close()
            if self.wandb_run is not None:
                self.wandb_run.finish()

        return best_ckpt_info

    def forward_pass(self, obs, policy):
        obs = dict_apply(
            obs,
            lambda x: x.cuda()
            if not isinstance(x, list)
            else [dict_apply(item, lambda y: y.cuda()) for item in x],
        )
        return policy(obs)

    def plot_history(
        self, train_history, validation_history, num_epochs, ckpt_dir, seed
    ):
        # save training curves
        for key in train_history[0]:
            plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
            plt.figure()
            train_values = [summary[key].item() for summary in train_history]
            val_values = [summary[key].item() for summary in validation_history]
            plt.plot(
                np.linspace(0, num_epochs - 1, len(train_history)),
                train_values,
                label="train",
            )
            plt.plot(
                np.linspace(0, num_epochs - 1, len(validation_history)),
                val_values,
                label="validation",
            )
            # plt.ylim([-0.1, 1])
            plt.tight_layout()
            plt.legend()
            plt.title(key)
            plt.savefig(plot_path)
        print(f"Saved plots to {ckpt_dir}")
