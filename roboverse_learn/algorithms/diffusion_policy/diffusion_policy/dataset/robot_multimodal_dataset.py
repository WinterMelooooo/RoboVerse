import copy
from typing import Dict

import numba
import numpy as np
import torch
from diffusion_policy.common.normalize_util import get_image_range_normalizer, get_identity_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from termcolor import cprint
from omegaconf import DictConfig
from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.dataset.robot_pointcloud_dataset import transform_point_cloud, ROBOT_ROOT_STATES
from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.dataset.robot_spUnet_dataset import point_collate_fn
from typing import Any, Dict, List
from roboverse_learn.algorithms.utils.transformpcd import ComposePCD

class MultiModalDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=64,
        max_train_episodes=None,
        max_visible_ratio=100,
        norm_pnt_cloud=False,
        transform_pcd: List[Dict[str, Any]] = None,
        n_obs_steps=2,
        pnt_cloud_with_extra=False,
        rgb_with_depth=False,
        cotraining=False,
        real_world_zarr_path=None,
        real_world_ratio=0.0,
    ):

        super().__init__()
        # cprint(zarr_path, "red")
        # cprint(batch_size, "red")
        self.replay_buffer:ReplayBuffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            keys=["head_camera", "state", "action", "head_camera_pnt_cloud", "head_camera_depth"],
        )
        print(f"Replay buffer size: {self.replay_buffer.n_episodes}")
        keep_n_episodes = self.replay_buffer.n_episodes * max_visible_ratio / 100.0
        while self.replay_buffer.n_episodes > keep_n_episodes:
            self.replay_buffer.pop_episode()
        if cotraining:
            assert real_world_zarr_path is not None and real_world_ratio > 0.0, "real_world_zarr_path must be provided for cotraining."
            self.replay_buffer.add_from_path(real_world_zarr_path)
            now_n_episodes = (self.replay_buffer.n_episodes - keep_n_episodes) * real_world_ratio / 100.0 + keep_n_episodes
            while self.replay_buffer.n_episodes > now_n_episodes:
                self.replay_buffer.pop_episode()
            print(f"Using {keep_n_episodes} simulation episodes and {self.replay_buffer.n_episodes - keep_n_episodes} real world episodes for training and validation.")
        else:
            print(f"Using {self.replay_buffer.n_episodes} episodes for training and validation.")

        val_mask = get_val_mask(n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
        )
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.norm_pnt_cloud = norm_pnt_cloud
        self.n_obs_steps = n_obs_steps
        self.grid_pnt_cloud = bool(transform_pcd is not None)
        self.batch_size = batch_size
        self.transform_pcd = ComposePCD(transform_pcd)
        self.name = zarr_path.split("/")[-1].split("_")[0]
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()
        self.pnt_cloud_with_extra = pnt_cloud_with_extra
        self.rgb_with_depth = rgb_with_depth
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode="limits", **kwargs):
        norm_dict = {
            "head_cam": get_image_range_normalizer,
            "front_cam": get_image_range_normalizer,
            "left_cam": get_image_range_normalizer,
            "right_cam": get_image_range_normalizer,
            #"coord": get_identity_normalizer,
            #"color": get_identity_normalizer,
            #"feat": get_identity_normalizer,
            #"offset": get_identity_normalizer,
        }
        fit_dict = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
        }
        if not self.norm_pnt_cloud:
            norm_dict["point_cloud"] = get_identity_normalizer
        else:
            fit_dict["point_cloud"] = self.replay_buffer["head_camera_pnt_cloud"]
        normalizer = LinearNormalizer()
        normalizer.fit(data=fit_dict, last_n_dims=1, mode=mode, **kwargs)
        for key, func in norm_dict.items():
            normalizer[key] = func()
        return normalizer


    def __len__(self) -> int:
        return len(self.sampler)


    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        if isinstance(idx, slice):
            raise NotImplementedError  # Specialized
        elif isinstance(idx, int):
            sample = self.sampler.sample_sequence(idx)
            sample = dict_apply(sample, torch.from_numpy)
            return sample
        elif isinstance(idx, np.ndarray):
            # print(idx, len(idx))
            # print(self.batch_size)
            assert len(idx) == self.batch_size
            for k, v in self.sampler.replay_buffer.items():
                batch_sample_sequence(
                    self.buffers[k],
                    v,
                    self.sampler.indices,
                    idx,
                    self.sampler.sequence_length,
                )
            return self.buffers_torch
        else:
            raise ValueError(idx)

    def postprocess(self, samples, device):
        agent_pos = samples["state"].to(device, non_blocking=True) # B, T, D
        head_cam = samples["head_camera"].to(device, non_blocking=True) / 255.0 # B, T, 3, H, W
        if self.rgb_with_depth:
            head_cam = self.add_depth(samples, head_cam, device) # B, T, 4, H, W
        action = samples["action"].to(device, non_blocking=True) # B, T, D
        data = self.add_pntcloud(samples, agent_pos, head_cam, action, device)
        data = dict_apply(data, lambda x: x.to(device, non_blocking=True))
        return data

    def add_pntcloud(self, samples, agent_pos, head_cam, action, device):
        point_cloud = samples["head_camera_pnt_cloud"].to(device, non_blocking=True)# B, T, 4096, 6
        if not self.pnt_cloud_with_extra:
            point_cloud = point_cloud[..., :3]# B, T, 4096, 3
        if not self.norm_pnt_cloud:
            # Transform the origin of the point cloud to robot root
            point_cloud = transform_point_cloud(point_cloud, ROBOT_ROOT_STATES, self.name, device)# B, T, 4096, 3
            last_dim = 3 if not self.pnt_cloud_with_extra else 8
            # if not (len(point_cloud.shape) == 4 and point_cloud.shape[2] == 4096 and point_cloud.shape[3] == last_dim):
            #     raise ValueError(f"point_cloud.shape = {point_cloud.shape}, while expecting to be (B, T, 4096, {last_dim})")


        if self.grid_pnt_cloud:
            B, T, N, C = point_cloud.shape
            point_cloud_batch = []
            for i in range(B):
                point_clouds = []
                masked_pnt_cloud = point_cloud[i, : self.n_obs_steps, :, :]
                for idx in range(self.n_obs_steps):
                    pntcloud = masked_pnt_cloud[idx].cpu().numpy()
                    coords = pntcloud[:, :3].astype(np.float32)
                    if self.pnt_cloud_with_extra:
                        colors = pntcloud[:, 3:6].astype(np.float32)
                        pcd_dict = self.transform_pcd({"coord": coords, "color": colors})
                    else:
                        pcd_dict = self.transform_pcd({"coord": coords})
                    point_clouds.append(pcd_dict)
                point_cloud_batch.append(point_clouds) # B, n_obs_steps, Dict[coord, color, feat, offset]
            flat_pcds = sum(point_cloud_batch, [])  # list of dict, length = B * n_obs_steps

            point_cloud = point_collate_fn(flat_pcds)
            # {
            #   'coord': Tensor[M,3],
            #   'grid_coord': Tensor[M,3],
            #   'feat': Tensor[M,F],
            #   'offset': Tensor[B*n_obs_steps]
            # }

        data =  {
            "obs": {
                "head_cam": head_cam,  # B, T, 3, H, W
                "agent_pos": agent_pos,  # B, T, D
                "point_cloud": point_cloud,  # B, T, 4096, 6 or B, T, Dict
            },
            "action": action,  # B, T, D
        }
        return data

    def add_depth(self, samples, head_cam, device):
        depth = samples["head_camera_depth"][:,:,:1,:,:].to(device, non_blocking=True) / 255.0 # (B, T, 4, H, W) [0,1]
        head_cam = torch.cat([head_cam, depth], dim=2)  # B, T, 4, H, W
        return head_cam

def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[idx[i]]
        data[i, sample_start_idx:sample_end_idx] = input_arr[buffer_start_idx:buffer_end_idx]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(_batch_sample_sequence, nopython=True, parallel=False)
_batch_sample_sequence_parallel = numba.jit(_batch_sample_sequence, nopython=True, parallel=True)


def batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    batch_size = len(idx)
    assert data.shape == (batch_size, sequence_length, *input_arr.shape[1:])
    if batch_size >= 16 and data.nbytes // batch_size >= 2**16:
        _batch_sample_sequence_parallel(data, input_arr, indices, idx, sequence_length)
    else:
        _batch_sample_sequence_sequential(data, input_arr, indices, idx, sequence_length)
