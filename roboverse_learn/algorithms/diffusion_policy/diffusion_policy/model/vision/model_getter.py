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
    import os
    import torch.nn as nn
    from importlib import import_module
    from collections import namedtuple
    from models.builder import EncoderDecoder as segmodel
    from .rgbd_encoder import DFormerFeatureExtractor
    Args = namedtuple("Args", ["syncbn", "compile", "continue_fpath", "sliding"])
    ckpt_path = "checkpoints/trained/NYUv2_" + name + ".pth" if not "v2" in name else "checkpoints/trained/" + name + "_NYU.pth"
    args = Args(syncbn=True, compile=False, sliding = False, continue_fpath=ckpt_path)
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

def get_resnet_rgbd(name, weights = None, **kwargs):
    import torch.nn as nn
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_pointnet(name, **kwargs):
    from .pointnet import PointNetfeat
    return PointNetfeat()


def get_vit(name, **kwargs):
    """
    name: vit_base_patch16, vit_large_patch16
    """
    try:
        from .vit import ViT
    except:
        import sys
        sys.path.append("./roboverse_learn/algorithms/diffusion_policy/diffusion_policy/model/vision")
        from vit import ViT
    vit = ViT(name, **kwargs).to(device="cpu")
    return vit

def get_multivit(ckpt_path):
    try:
        from .multivit import MultiViTModel
    except:
        import sys
        sys.path.append("./roboverse_learn/algorithms/diffusion_policy/diffusion_policy/model/vision")
        from multivit import MultiViTModel
    multimit = MultiViTModel(ckpt_path).to(device="cpu")
    return multimit


def get_spUnet(**kwargs):
    try:
        from .spUnet import SpUNet
    except:
        import sys
        sys.path.append(".")
        from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.model.vision.spUnet import SpUNet
    return SpUNet(**kwargs)
