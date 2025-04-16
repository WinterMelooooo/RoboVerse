import hydra
from diffusion_policy.dataset.robot_pointcloud_dataset import RobotPointCloudDataset
from omegaconf import OmegaConf
import rootutils
import pathlib
from torch.utils.data import DataLoader
import numpy as np
import torch
rootutils.setup_root(__file__, pythonpath=True)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


abs_config_path = str(pathlib.Path(__file__).resolve().parent.parent.joinpath("diffusion_policy", "config").absolute())




class BatchSampler:
    def __init__(
        self,
        data_size: int,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = True,
    ):
        assert drop_last
        self.data_size = data_size
        self.batch_size = batch_size
        self.num_batch = data_size // batch_size
        self.discard = data_size - batch_size * self.num_batch
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed) if shuffle else None

    def __iter__(self):
        if self.shuffle:
            perm = self.rng.permutation(self.data_size)
        else:
            perm = np.arange(self.data_size)
        if self.discard > 0:
            perm = perm[: -self.discard]
        perm = perm.reshape(self.num_batch, self.batch_size)
        for i in range(self.num_batch):
            yield perm[i]

    def __len__(self):
        return self.num_batch


def create_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    seed: int = 0,
):
    # print("create_dataloader_batch_size", batch_size)
    batch_sampler = BatchSampler(len(dataset), batch_size, shuffle=shuffle, seed=seed, drop_last=True)

    def collate(x):
        assert len(x) == 1
        return x[0]

    dataloader = DataLoader(
        dataset,
        collate_fn=collate,
        sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=persistent_workers,
    )
    return dataloader
@hydra.main(
    version_base=None,
    config_path=abs_config_path,
)
def main(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    dataset: RobotPointCloudDataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    #print("dataset_path", cfg.task.dataset.zarr_path)
    train_dataloader = create_dataloader(dataset, **cfg.dataloader)
    device = torch.device(cfg.training.device)
    #print("device", device)
    for batch in train_dataloader:
        batch = dataset.postprocess(batch, device)
        ranidx = np.random.randint(0, len(batch["obs"]["point_cloud"]))
        ranobs = np.random.randint(0, len(batch["obs"]["point_cloud"][0]))
        pnt_cloud = batch["obs"]["point_cloud"][ranidx,ranobs].cpu()
        print(pnt_cloud.shape)
        import sys
        sys.path.append(".")
        import roboverse_learn.algorithms.utils.visualizer.visualizer as visualizer
        visualizer.visualize_pointcloud(pnt_cloud)
        break


if __name__ == "__main__":
    main()
