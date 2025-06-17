from omegaconf import OmegaConf
import hydra
import pathlib
import torch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import yaml
import json
import wandb
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
from roboverse_learn.algorithms.act.common.json_logger import JsonLogger

from .constants import DT
from .constants import PUPPET_GRIPPER_JOINT_OPEN
from .utils import load_data  # data functions
from .utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from .policy import ACTPolicy, CNNMLPPolicy
from roboverse_learn.algorithms.act.workspace.base_workspace import BaseWorkspace
import IPython
e = IPython.embed
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

class RobotWorkSpace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]
    def __init__(
        self, cfg: OmegaConf, output_dir=None, local_rank=None, world_size=None
    ):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = world_size
        OmegaConf.set_struct(cfg.logging, False)
        self.logger = cfg.logging.pop("logger_name", "wandb")
        OmegaConf.set_struct(cfg.logging, True)
        super().__init__(cfg, output_dir=output_dir)
        seed = cfg.seed
        set_seed(seed)
        # command line parameters
        policy_class = cfg.policy_class
        task_name = cfg.task_name
        batch_size_train = cfg.batch_size
        batch_size_val = cfg.batch_size
        num_epochs = cfg.num_epochs

        # get task parameters
        dataset_dir = cfg.dataset_dir
        num_episodes = cfg.num_episodes
        episode_len = cfg.episode_len
        camera_names = cfg.camera_names

        # fixed parameters
        lr_backbone = 1e-5
        backbone = 'resnet18'
        if policy_class == 'ACT':
            enc_layers = 4
            dec_layers = 7
            nheads = 8
            policy_config = {'lr': cfg.lr,
                            'num_queries': cfg.chunk_size,
                            'kl_weight': cfg.kl_weight,
                            'hidden_dim': cfg.hidden_dim,
                            'dim_feedforward': cfg.dim_feedforward,
                            'lr_backbone': lr_backbone,
                            'backbone': backbone,
                            'enc_layers': enc_layers,
                            'dec_layers': dec_layers,
                            'nheads': nheads,
                            'camera_names': camera_names,
                            'state_dim': cfg.state_dim
                            }
        elif policy_class == 'CNNMLP':
            policy_config = {'lr': cfg.lr, 'lr_backbone': lr_backbone, 'backbone': backbone, 'num_queries': 1,
                            'camera_names': camera_names, }
        else:
            raise NotImplementedError

        ckpt_dir = f"info/outputs/ACT/{datetime.now().strftime('%Y.%m.%d')}/{datetime.now().strftime('%H.%M.%S')}_{task_name}_{num_episodes}"

        # Load metadata from dataset directory
        metadata_path = os.path.join(dataset_dir, 'metadata.json')
        dataset_metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                dataset_metadata = json.load(f)

        config = {
            'num_epochs': num_epochs,
            'ckpt_dir': ckpt_dir,
            'episode_len': episode_len,
            'lr': cfg.lr,
            'policy_class': policy_class,
            'policy_config': policy_config,
            'task_name': task_name,
            'seed': cfg.seed,
            'temporal_agg': cfg.temporal_agg,
            'camera_names': camera_names,
            'real_robot': True,
            'data': dataset_metadata,  # Add the dataset metadata to config
        }

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
                    **cfg.logging,   # project/name/tags/mode/id/group 等
                )
                wandb.config.update({"output_dir": self.output_dir})

        train_dataloader, val_dataloader, stats, _ = load_data(cfg.dataset_dir, cfg.num_episodes, cfg.camera_names, cfg.batch_size_train, cfg.batch_size_val, seed=cfg.seed)

        # save dataset stats
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        stats_path = os.path.join(self.output_dir, f'dataset_stats.pkl')
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)

        best_ckpt_info = self.train_bc(train_dataloader, val_dataloader)
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info

        # save best checkpoint
        ckpt_path = os.path.join(os.path.join(self.output_dir, "checkpoints"), f'policy_best.ckpt')
        torch.save(best_state_dict, ckpt_path)
        print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


    def train_bc(self, train_dataloader, val_dataloader):
        local_rank = self.local_rank
        config = self.cfg
        num_epochs = config.training.num_epochs
        seed = config.training.seed

        policy = hydra.utils.instantiate(config.policy)
        policy.cuda()
        optimizer = policy.configure_optimizers()
        policy = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        min_val_loss = np.inf
        best_ckpt_info = None
        if local_rank == 0:
            epoch_iter = tqdm(range(num_epochs), desc="Epoch")
        else:
            epoch_iter = range(num_epochs)
        global_step = 0
        for epoch in epoch_iter:
            if local_rank == 0:
                epoch_iter.set_description(f"Epoch {epoch+1}/{num_epochs}")
            train_dataloader.sampler.set_epoch(epoch)
            val_dataloader.sampler.set_epoch(epoch)

            # training
            policy.train()
            optimizer.zero_grad()
            train_losses = []
            for batch_idx, data in enumerate(train_dataloader):
                forward_dict = self.forward_pass(data, policy)
                # backward
                loss = forward_dict['loss']
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                step_log = {
                    "train_loss": loss.item(),
                    "global_step": global_step,
                    "epoch": epoch,
                    # "lr": lr_scheduler.get_last_lr()[0],
                }
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
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = self.forward_pass(data, policy)
                    epoch_dicts.append(forward_dict)
                epoch_summary = compute_dict_mean(epoch_dicts)
                epoch_val_loss = epoch_summary['loss']
                if self.local_rank == 0:
                    step_log["val_loss"] = epoch_val_loss

                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

            # End of epoch
            if local_rank == 0:
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss
                epoch_iter.set_postfix(loss=train_loss.item(), refresh=False)

        if self.local_rank == 0:
            # json_logger.close()
            if self.wandb_run is not None:
                self.wandb_run.finish()
        if (((self.epoch + 1) % cfg.training.checkpoint_every) == 0 or (self.epoch + 1) == cfg.training.num_epochs) \
            and local_rank == 0:
            self.save_checkpoint(f"epoch_{self.epoch+1}.ckpt")

        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
        if local_rank == 0:
            torch.save(best_state_dict, ckpt_path)
            print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')
            if self.wandb_run is not None:
                self.wandb_run.finish()


        return best_ckpt_info



    def forward_pass(self, data, policy):
        image_data, qpos_data, action_data, is_pad = data
        image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
        return policy(qpos_data, image_data, action_data, is_pad)


    def plot_history(self, train_history, validation_history, num_epochs, ckpt_dir, seed):
        # save training curves
        for key in train_history[0]:
            plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
            plt.figure()
            train_values = [summary[key].item() for summary in train_history]
            val_values = [summary[key].item() for summary in validation_history]
            plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
            plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
            # plt.ylim([-0.1, 1])
            plt.tight_layout()
            plt.legend()
            plt.title(key)
            plt.savefig(plot_path)
        print(f'Saved plots to {ckpt_dir}')
