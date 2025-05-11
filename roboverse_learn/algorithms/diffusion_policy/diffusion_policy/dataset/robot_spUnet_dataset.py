import copy
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List
import clip

import numba
import numpy as np
import torch
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from termcolor import cprint
from torch.utils.data import default_collate

sys.path.append(".")
from roboverse_learn.algorithms.utils.transformpcd import ComposePCD

VARIATION_DESCRIPTION = {
    "CloseBox": ['close box',
                'close the lid on the box',
                'shut the box',
                'shut the box lid'],
}
def get_task_name(task_name):
    for name in VARIATION_DESCRIPTION.keys():
        if task_name in name or name in task_name:
            return name
    raise ValueError(
        f"task_name {task_name} not in {list(VARIATION_DESCRIPTION.keys())}"
    )

class RobotPointCloudDataset(BaseImageDataset):
    def __init__(
        self,
        zarr_path,
        task_name,
        horizon=1,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        batch_size=64,
        max_train_episodes=None,
        max_visible_ratio=100,
        transform_pcd: List[Dict[str, Any]] = None,
        n_obs_steps=2,
        shape_meta=None,
    ):
        super().__init__()
        # cprint(zarr_path, "red")
        # cprint(batch_size, "red")
        task_name = get_task_name(task_name)
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            # keys=['head_camera', 'front_camera', 'left_camera', 'right_camera', 'state', 'action'],
            keys=["head_camera", "state", "action", "head_camera_pnt_cloud"],
        )
        print(f"Replay buffer size: {self.replay_buffer.n_episodes}")
        keep_n_episodes = self.replay_buffer.n_episodes * max_visible_ratio / 100.0
        while self.replay_buffer.n_episodes > keep_n_episodes:
            self.replay_buffer.pop_episode()
        print(
            f"Using {self.replay_buffer.n_episodes} episodes for training and validation."
        )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )

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
        self.n_obs_steps = n_obs_steps

        self.batch_size = batch_size
        sequence_length = self.sampler.sequence_length
        self.buffers = {
            k: np.zeros((batch_size, sequence_length, *v.shape[1:]), dtype=v.dtype)
            for k, v in self.sampler.replay_buffer.items()
        }
        self.buffers_torch = {k: torch.from_numpy(v) for k, v in self.buffers.items()}
        for v in self.buffers_torch.values():
            v.pin_memory()

        self.transform_pcd = ComposePCD(transform_pcd)
        if shape_meta is not None and "goal" in shape_meta.keys() and shape_meta["goal"] is not None:

            clip_model = "ViT-B/16"
            clip_model, _ = clip.load(
                clip_model, device="cuda", download_root=os.path.expanduser("~/yktang/.cache/clip")
            )
            clip_model.requires_grad_(False)
            clip_model.eval()
            description_token = clip.tokenize(VARIATION_DESCRIPTION[task_name][0]).to("cuda")
            task_goal = clip_model.encode_text(description_token).cpu().numpy()
            self.task_goal = dict(task_emb=task_goal.reshape(-1))
            print(f"Successfully encoded task goal: {VARIATION_DESCRIPTION[task_name]}")

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
        data = {
            "action": self.replay_buffer["action"],
            "qpos": self.replay_buffer["state"],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer["qpos"] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample["state"].astype(np.float32)  # (agent_posx2, block_posex3)
        head_cam = np.moveaxis(sample["head_camera"], -1, 1) / 255.0
        point_cloud = sample["head_camera_pnt_cloud"][:,]  # (T, 1024, 6)
        assert (
            len(point_cloud.shape) == 3
            and point_cloud.shape[2] == 6
            or point_cloud.shape[2] == 3
        ), (
            f"point_cloud.shape = {point_cloud.shape}, while expecting to be (T, 1024, 6) or (T, 1024, 3)"
        )
        data = {
            "obs": {
                "head_cam": head_cam,  # T, 3, H, W
                "agent_pos": agent_pos,  # T, D
                "point_cloud": point_cloud,  # T, 1024, 6
            },
            "action": sample["action"].astype(np.float32),  # T, D
        }
        return data

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
            assert len(idx) == self.batch_size, (
                f"len(idx) = {len(idx)}, while expecting to be {self.batch_size}"
            )
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
        agent_pos = samples["state"].to(device, non_blocking=True)
        action = samples["action"].to(device, non_blocking=True)
        point_cloud = samples["head_camera_pnt_cloud"].to(device, non_blocking=True)
        B, T, N, C = point_cloud.shape

        point_cloud_batch = []
        for i in range(B):
            point_clouds = []
            masked_pnt_cloud = point_cloud[i, : self.n_obs_steps, :, :]
            for idx in range(self.n_obs_steps):
                pntcloud = masked_pnt_cloud[idx].cpu().numpy()
                coords = pntcloud[:, :3].astype(np.float32)
                colors = pntcloud[:, 3:6].astype(np.float32)
                pcd_dict = self.transform_pcd({"coord": coords, "color": colors})
                point_clouds.append(pcd_dict)
            point_cloud_batch.append(point_clouds)
        data = {
            "obs": {
                "qpos": agent_pos,  # B, T, D
                "pcds": point_cloud_batch,  # B, n_obs_steps, Dict[coord, color, feat, offset]
            },
            "action": action,  # B, T, D
        }
        flat_pcds = sum(
            data["obs"]["pcds"], []
        )  # list of dict, length = B * n_obs_steps

        collated = point_collate_fn(flat_pcds)
        # {
        #   'coord': Tensor[M,3],
        #   'grid_coord': Tensor[M,3],
        #   'feat': Tensor[M,F],
        #   'offset': Tensor[B*n_obs_steps]
        # }
        data["obs"]["pcds"] = collated
        if hasattr(self, "task_goal"):
            goal_list = [self.task_goal] * B
            # collate 后得到 {'task_emb': Tensor(B, C)}
            data["goal"] = default_collate(goal_list)
        data = dict_apply(data, lambda x: x.to(device, non_blocking=True))
        return data


def _batch_sample_sequence(
    data: np.ndarray,
    input_arr: np.ndarray,
    indices: np.ndarray,
    idx: np.ndarray,
    sequence_length: int,
):
    for i in numba.prange(len(idx)):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = indices[
            idx[i]
        ]
        data[i, sample_start_idx:sample_end_idx] = input_arr[
            buffer_start_idx:buffer_end_idx
        ]
        if sample_start_idx > 0:
            data[i, :sample_start_idx] = data[i, sample_start_idx]
        if sample_end_idx < sequence_length:
            data[i, sample_end_idx:] = data[i, sample_end_idx - 1]


_batch_sample_sequence_sequential = numba.jit(
    _batch_sample_sequence, nopython=True, parallel=False
)
_batch_sample_sequence_parallel = numba.jit(
    _batch_sample_sequence, nopython=True, parallel=True
)


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
        _batch_sample_sequence_sequential(
            data, input_arr, indices, idx, sequence_length
        )


def point_collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [point_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: point_collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)
