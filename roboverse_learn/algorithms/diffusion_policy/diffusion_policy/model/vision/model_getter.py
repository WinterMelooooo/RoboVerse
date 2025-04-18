import torch
import torchvision


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
    import torch.nn as nn
    from importlib import import_module
    from collections import namedtuple
    from models.builder import EncoderDecoder as segmodel
    from .rgbd_encoder import DFormerFeatureExtractor
    Args = namedtuple("Args", ["syncbn", "compile", "continue_fpath", "sliding"])
    args = Args(syncbn=True, compile=False, sliding = False, continue_fpath="checkpoints/trained/NYUv2_DFormer_Base.pth")

    cfg_path = "local_configs.NYUDepthv2." + name
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
    device = torch.device("cpu")
    model.to(device)
    return DFormerFeatureExtractor(model.backbone)
