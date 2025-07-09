import types
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torchvision
from termcolor import cprint


def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    # resnet_new = torch.nn.Sequential(
    #     resnet,
    #     torch.nn.Linear(512, 128)
    # )
    # return resnet_new
    return resnet


def get_resnet_pixelwise(name, weights=None, **kwargs):
    resnet = get_resnet(name, weights=weights, **kwargs)
    resnet.avgpool = nn.Identity()

    # 定义一个“去掉 flatten”的新 _forward_impl
    def new_forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return self.fc(x)

    resnet._forward_impl = types.MethodType(new_forward_impl, resnet)
    return resnet


def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m

    r3m.device = "cpu"
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to("cpu")
    return resnet_model


def get_dformer(name, **kwargs):
    """
    name: DFormer_Large, DFormer_Small, DFormer_Base, DFormer_Tiny, DFormerv2_Large, DFormerv2_Small, DFormerv2_Base
    Forward:
        Args:
            x: (B, 3, H, W) tensor, representing rgb image in BGR format
            modal_x: (B, 3, H, W) tensor, representing depth image in BGR format
    """
    import sys

    sys.path.append("./third_party/DFormer")
    import os
    from collections import namedtuple
    from importlib import import_module

    import torch.nn as nn
    from models.builder import EncoderDecoder as segmodel

    from .rgbd_encoder import DFormerFeatureExtractor

    Args = namedtuple("Args", ["syncbn", "compile", "continue_fpath", "sliding"])
    ckpt_path = (
        "checkpoints/trained/NYUv2_" + name + ".pth" if not "v2" in name else "checkpoints/trained/" + name + "_NYU.pth"
    )
    args = Args(syncbn=True, compile=False, sliding=False, continue_fpath=ckpt_path)
    cfg_path = "local_configs.NYUDepthv2." + name if not "v2" in name else "local_configs.NYUDepthv2." + name[:11]
    print(f"Loading ckpt_path: {ckpt_path}")
    print(f"Loading cfg_path: {cfg_path}")
    config = getattr(import_module(cfg_path), "C")
    criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)
    BatchNorm2d = nn.SyncBatchNorm if args.syncbn else nn.BatchNorm2d

    model = segmodel(
        cfg=config,
        criterion=criterion,
        norm_layer=BatchNorm2d,
        syncbn=args.syncbn,
    )
    weight = torch.load(args.continue_fpath, map_location=torch.device("cpu"))
    print(f"Loading weights from {args.continue_fpath}")
    if "model" in weight:
        weight = weight["model"]
    elif "state_dict" in weight:
        weight = weight["state_dict"]
    print(model.load_state_dict(weight, strict=False))
    if not args.syncbn:
        device = torch.device("cpu")
    else:
        try:
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        except:
            device = torch.device("cuda")
    model.to(device)
    return DFormerFeatureExtractor(model.backbone, device=device)


def get_resnet_rgbd(name, weights=None, **kwargs):
    import torch.nn as nn

    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = torch.nn.Identity()
    return resnet


def get_pointnet(**kwargs):
    from .dp3_encoder import PointNetEncoderXYZ

    return PointNetEncoderXYZ(**kwargs)


def get_pointnet_pointwise(pre_norm=True, **kwargs):
    pntnet = get_pointnet(**kwargs)
    pntnet.pool = nn.Identity()
    pntnet.final_projection = nn.LayerNorm(pntnet.block_channel[-1]) if pre_norm else nn.Identity()
    return pntnet


def get_vit(name, **kwargs):
    """
    name: vit_base_patch16, vit_large_patch16
    """
    try:
        from .vit import ViT
    except:
        import sys

        sys.path.append("./roboverse_learn/algorithms/mlp_policy/mlp_policy/model/vision")
        from vit import ViT
    vit = ViT(name, **kwargs).to(device="cpu")
    return vit


def get_multivit(ckpt_path):
    try:
        from .multivit import MultiViTModel
    except:
        import sys

        sys.path.append("./roboverse_learn/algorithms/mlp_policy/mlp_policy/model/vision")
        from multivit import MultiViTModel
    multimit = MultiViTModel(ckpt_path).to(device="cpu")
    return multimit


def get_spUnet(**kwargs):
    try:
        from .spUnet import SpUNet
    except:
        import sys

        sys.path.append(".")
        from roboverse_learn.algorithms.mlp_policy.mlp_policy.model.vision.spUnet import SpUNet
    return SpUNet(**kwargs)


def get_spUnet_encoder(**kwargs):
    """
    Returns a SpUNet encoder that outputs a feature map of shape (B, C, H, W)
    """
    try:
        from .spUnet import SpUnetEncoder
    except:
        import sys

        sys.path.append(".")
        from roboverse_learn.algorithms.mlp_policy.mlp_policy.model.vision.spUnet import SpUNetEncoder
    return SpUnetEncoder(**kwargs)


def get_prediction_mlp(**kawrgs):
    try:
        from .sensor_prediction_mlp import SensorPredictor
    except:
        import sys

        sys.path.append(".")
        from roboverse_learn.algorithms.mlp_policy.mlp_policy.model.vision.sensor_prediction_mlp import SensorPredictor
    return SensorPredictor(**kawrgs)


def get_prediction_mlp_with_attention(**kawrgs):
    try:
        from .sensor_prediction_mlp import SensorPredictorCLS
    except:
        import sys

        sys.path.append(".")
        from roboverse_learn.algorithms.mlp_policy.mlp_policy.model.vision.sensor_prediction_mlp import (
            SensorPredictorCLS,
        )
    return SensorPredictorCLS(**kawrgs)


def get_prediction_transformer(**kawrgs):
    try:
        from .sensor_prediction_mlp import TransSensorPredictor
    except:
        import sys

        sys.path.append(".")
        from roboverse_learn.algorithms.mlp_policy.mlp_policy.model.vision.sensor_prediction_mlp import (
            TransSensorPredictor,
        )
    return TransSensorPredictor(**kawrgs)


def get_state_mlp(
    observation_space: Dict,
    state_mlp_size=(64, 64),
    state_mlp_activation_fn=nn.ReLU,
):
    state_key = "agent_pos"
    state_shape = observation_space[state_key]["shape"]
    if len(state_mlp_size) == 0:
        raise RuntimeError(f"State mlp size is empty")
    elif len(state_mlp_size) == 1:
        net_arch = []
    else:
        net_arch = state_mlp_size[:-1]
    output_dim = state_mlp_size[-1]
    state_mlp = nn.Sequential(*create_mlp(state_shape[0], output_dim, net_arch, state_mlp_activation_fn))
    return state_mlp


def get_sensor_mlp(
    observation_space: Dict,
    state_mlp_size=(64, 64),
    state_mlp_activation_fn=nn.ReLU,
):
    state_key = "franka_panda_leftfinger_touch_sensor_pred"
    state_shape = observation_space[state_key]["shape"]
    if len(state_mlp_size) == 0:
        raise RuntimeError(f"State mlp size is empty")
    elif len(state_mlp_size) == 1:
        net_arch = []
    else:
        net_arch = state_mlp_size[:-1]
    output_dim = state_mlp_size[-1]
    state_mlp = nn.Sequential(*create_mlp(state_shape[0], output_dim, net_arch, state_mlp_activation_fn))
    return state_mlp


def get_late_fusion_resnet_dp3(**kawrgs):
    try:
        from .dp3_resnet_fusion_encoder import FusionLateEncoder
    except:
        import sys

        sys.path.append(".")
        from roboverse_learn.algorithms.mlp_policy.mlp_policy.model.vision.dp3_resnet_fusion_encoder import (
            FusionLateEncoder,
        )
    return FusionLateEncoder(**kawrgs)


def get_early_fusion_resnet_dp3(**kawrgs):
    try:
        from .dp3_resnet_fusion_encoder import FusionEarlyEncoder
    except:
        import sys

        sys.path.append(".")
        from roboverse_learn.algorithms.mlp_policy.mlp_policy.model.vision.dp3_resnet_fusion_encoder import (
            FusionEarlyEncoder,
        )
    return FusionEarlyEncoder(**kawrgs)


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


def get_identity():
    return nn.Identity()
