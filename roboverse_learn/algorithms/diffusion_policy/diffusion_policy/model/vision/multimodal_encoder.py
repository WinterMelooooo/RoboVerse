import copy
import os
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torchvision
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
IMG_OBS_TYPES = ["rgb", "rgbd_resnet"]
PNTCLOUD_OBS_TYPES = ["point_cloud"]
ROBOT_STATE_OBS_TYPES = ["low_dim"]

class ImgEncoderArgs():
    def __init__(self,
                 resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
                 crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
                 random_crop: bool = True,
                 use_group_norm: bool = False,
                 imagenet_norm: bool = False,
                ):
        self.resize_shape = resize_shape
        self.crop_shape = crop_shape
        self.random_crop = random_crop
        self.use_group_norm = use_group_norm
        self.imagenet_norm = imagenet_norm

class PntCloudEncoderArgs():
    def __init__(self,params):
        pass

class RobotStateEncoderArgs():
    def __init__(self,params):
        pass




class MultiModalEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        img_model: Union[nn.Module, Dict[str, nn.Module]],
        pntcloud_model: Union[nn.Module, Dict[str, nn.Module]],
        robot_state_model: Union[nn.Module, Dict[str, nn.Module]],
        img_encoder_args:ImgEncoderArgs = None,
        pntcloud_encoder_args:PntCloudEncoderArgs = None,
        robot_state_encoder_args:RobotStateEncoderArgs = None,
        fusion_args = None,
    ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        img_keys = list()
        low_dim_keys = list()
        point_cloud_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()


        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if type == "rgb" or type == "rgbd_resnet":
                img_keys.append(key)
                this_model = None
                if isinstance(img_model, dict):
                    this_model = img_model[key]
                else:
                    assert isinstance(img_model, nn.Module)
                    this_model = copy.deepcopy(img_model)

                if this_model is not None:
                    if img_encoder_args.use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model
                key_transform_map[key] = self._impl_img_transform(shape, type, key, img_encoder_args)
            elif type == "low_dim":
                low_dim_keys.append(key)
                this_model = copy.deepcopy(robot_state_model)
                key_model_map[key] = this_model
                key_transform_map[key] = nn.Identity()
            elif type == "point_cloud":
                point_cloud_keys.append(key)
                # configure model for this key
                this_model = copy.deepcopy(pntcloud_model)
                key_model_map[key] = this_model
                key_transform_map[key] = nn.Identity()

            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        img_keys = sorted(img_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.img_keys = img_keys
        self.low_dim_keys = low_dim_keys
        self.point_cloud_keys = point_cloud_keys
        self.key_shape_map = key_shape_map
        self.fusion_func = self._get_fusion_func(fusion_args)
        try:
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        except Exception as e:
            device = torch.device("cuda")
        self.to(device)


    def forward(self, obs_dict):
        batch_size = None
        batch_size_pnt = None
        img_features = list()
        low_dim_features = list()
        pntcloud_features = list()
        try:
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        except Exception as e:
            device = torch.device("cuda")
        # run each rgb obs to independent models
        for key in self.img_keys:
            img = obs_dict[key]
            if batch_size is None:
                batch_size = img.shape[0]
            else:
                assert batch_size == img.shape[0]
            assert img.shape[1:] == self.key_shape_map[key], (
                f"{img.shape} vs {self.key_shape_map[key]}"
            )
            # print(f"{key}: {img.shape}")
            img = self.key_transform_map[key](img)
            feature = self.key_model_map[key](img)
            img_features.append(feature.to(device))
            # print(f"{key}: {features[-1].device}")
        # process lowdim input
        for key in self.low_dim_keys:
            state = obs_dict[key]
            if batch_size is None:
                batch_size = state.shape[0]
            else:
                assert batch_size == state.shape[0]
            assert state.shape[1:] == self.key_shape_map[key]
            state = self.key_transform_map[key](state)
            feature = self.key_model_map[key](state)
            low_dim_features.append(feature.to(device))
            # print(f"{key}: {features[-1].device}")

        for key in self.point_cloud_keys:
            pnt_cloud = obs_dict[key]
            if batch_size_pnt is None:
                if isinstance(pnt_cloud, Dict):
                    first_key = next(iter(pnt_cloud))
                    batch_size_pnt = pnt_cloud[first_key].shape[0]
                else:
                    batch_size_pnt = pnt_cloud.shape[0]
            else:
                if isinstance(pnt_cloud, Dict):
                    first_key = next(iter(pnt_cloud))
                    assert batch_size_pnt == len(pnt_cloud[first_key]), f"batch_size_pnt mismatch for key {key}: {batch_size_pnt} vs {len(pnt_cloud[first_key])}"
                else:
                    assert batch_size_pnt == pnt_cloud.shape[0]
            #assert pnt_cloud.shape[1:] == self.key_shape_map[key], f"pnt_cloud.shape: {pnt_cloud.shape}, expected: {self.key_shape_map[key]}"
            pnt_cloud = self.key_transform_map[key](pnt_cloud)
            feature = self.key_model_map[key](pnt_cloud)
            pntcloud_features.append(feature.to(device))
            # print(f"{key}: {features[-1].device}")

        # concatenate all features
        result = self.fusion_func(img_features=img_features, low_dim_features=low_dim_features, pntcloud_features=pntcloud_features)
        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros(
                (batch_size,) + shape, dtype=self.dtype, device=self.device
            )
            example_obs_dict[key] = this_obs
        was_training = self.training
        self.eval()
        with torch.no_grad():
            example_output = self.forward(example_obs_dict)
        if was_training:
            self.train()
        output_shape = example_output.shape[1:]
        return output_shape


    def _impl_img_transform(self, shape, type, key, args):
        # configure resize
        input_shape = shape
        this_resizer = nn.Identity()
        if args.resize_shape is not None:
            if isinstance(args.resize_shape, dict):
                h, w = args.esize_shape[key]
            else:
                h, w = args.resize_shape
            this_resizer = torchvision.transforms.Resize(size=(h, w))
            input_shape = (shape[0], h, w)

        # configure randomizer
        this_randomizer = nn.Identity()
        if args.crop_shape is not None:
            if isinstance(args.crop_shape, dict):
                h, w = args.crop_shape[key]
            else:
                h, w = args.crop_shape
            if args.random_crop:
                this_randomizer = CropRandomizer(
                    input_shape=input_shape,
                    crop_height=h,
                    crop_width=w,
                    num_crops=1,
                    pos_enc=False,
                )
            else:
                this_randomizer = torchvision.transforms.CenterCrop(size=(h, w))
        # configure normalizer
        this_normalizer = nn.Identity()
        if args.imagenet_norm:
            if type == "rgb":
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]
            elif type == "rgbd_resnet":
                mean = [0.485, 0.456, 0.406, 0.308]
                std = [0.229, 0.224, 0.225, 0.299]
            this_normalizer = torchvision.transforms.Normalize(
                mean=[2 * x - 1 for x in mean],
                std=[2 * x for x in std],
            )
            print(
                f"{key} mean: {[2 * x - 1 for x in mean]}, std: {[2 * x for x in std]}"
            )
        this_transform = nn.Sequential(
            this_resizer, this_randomizer, this_normalizer
        )
        return this_transform


    def _get_fusion_func(self, fusion_args):
        fusion_method = fusion_args.fusion_method
        if fusion_method == "cat":
            return self._cat_features
        elif fusion_method == "cross_attention":
            embed_dim = fusion_args.embed_dim
            num_heads = fusion_args.num_heads
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
            self.img_proj = nn.Linear(fusion_args.img_dim, embed_dim)
            self.pc_proj = nn.Linear(fusion_args.pc_dim, embed_dim)
            self.state_proj = nn.Linear(fusion_args.state_dim, embed_dim)
            return self._cross_attention_features
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")


    def _cross_attention_features(self, img_features, low_dim_features, pntcloud_features):
        img_feats = [self.img_proj(f) for f in img_features]
        pc_feats = [self.pc_proj(f) for f in pntcloud_features]
        low_dim_feats = [self.state_proj(f) for f in low_dim_features]
        features = img_feats + low_dim_feats + pc_feats
        dims = [f.shape[-1] for f in features]
        assert len(set(dims)) == 1, "All feature dims must be equal for cross-attention"
        feat_stack = torch.stack(features, dim=0)
        attn_output, _ = self.cross_attn(feat_stack, feat_stack, feat_stack)
        fused = attn_output.mean(dim=0)
        return fused


    def _cat_features(self, img_features, low_dim_features, pntcloud_features):
        """
        Concatenate all features along the last dimension.
        """
        all_features = img_features + low_dim_features + pntcloud_features
        return torch.cat(all_features, dim=-1)
