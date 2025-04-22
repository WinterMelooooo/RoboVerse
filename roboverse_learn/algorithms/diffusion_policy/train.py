import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import pathlib

import hydra
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf

import rootutils
rootutils.setup_root(__file__, pythonpath=True)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch


abs_config_path = str(pathlib.Path(__file__).resolve().parent.joinpath("diffusion_policy", "config").absolute())

@hydra.main(
    version_base=None,
    config_path=abs_config_path,
)
def main(cfg: OmegaConf):

    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="Gloo")

    output_dir = cfg.training.output_dir if cfg.training.output_dir else None    
    workspace: BaseWorkspace = cls(cfg, local_rank=local_rank, world_size=world_size, output_dir=output_dir)
    workspace.run()


if __name__ == "__main__":
    main()
