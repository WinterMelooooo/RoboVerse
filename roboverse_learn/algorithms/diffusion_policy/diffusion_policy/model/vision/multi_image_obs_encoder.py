import copy
import os
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torchvision
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer

dic_means = {"CloseBox": 0.308515, "StackCube": 0.300145}
dic_stds = {"CloseBox": 0.299096, "StackCube": 0.303170}


class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str, nn.Module]],
        resize_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        crop_shape: Union[Tuple[int, int], Dict[str, tuple], None] = None,
        random_crop: bool = True,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool = False,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm: bool = False,
        test_rescale: bool = False,
        name=None,
    ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map["rgb"] = rgb_model

        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if type == "rgb":
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
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
                if imagenet_norm:
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]
                    if not test_rescale:
                        this_normalizer = torchvision.transforms.Normalize(
                            mean=mean, std=std
                        )
                        print(f"{key} mean: {mean}, std: {std}")
                    else:
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
                key_transform_map[key] = this_transform
            elif type == "low_dim":
                low_dim_keys.append(key)
            elif type == "rgbd":
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    key_model_map[key] = this_model
                from .preproccess import Preprocessor

                self.preprocessor = Preprocessor(
                    norm_mean=[0.485, 0.456, 0.406],
                    norm_std=[0.229, 0.224, 0.225],
                    device=this_model.device,
                )

            elif type == "rgbd_resnet":
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(size=(h, w))
                    input_shape = (shape[0], h, w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
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
                if imagenet_norm:
                    mean, std = None, None
                    for task_name in dic_means.keys():
                        if task_name in name:
                            mean = dic_means[task_name]
                            std = dic_stds[task_name]
                            break
                    if mean is None or std is None:
                        raise ValueError(f"Invalid name: {name}")
                    mean = [0.485, 0.456, 0.406, mean]
                    std = [0.229, 0.224, 0.225, std]
                    if not test_rescale:
                        this_normalizer = torchvision.transforms.Normalize(
                            mean=mean, std=std
                        )
                        print(f"{task_name} mean: {mean}, std: {std}")
                    else:
                        this_normalizer = torchvision.transforms.Normalize(
                            [2 * x - 1 for x in mean], [2 * x for x in std]
                        )
                        print(
                            f"{task_name} mean: {[2 * x - 1 for x in mean]}, std: {[2 * x for x in std]}"
                        )

                this_transform = nn.Sequential(
                    this_resizer, this_randomizer, this_normalizer
                )
                key_transform_map[key] = this_transform
            elif type == "point_cloud":
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features // 16,
                                num_channels=x.num_features,
                            ),
                        )
                    key_model_map[key] = this_model

                key_transform_map[key] = nn.Identity()

            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        try:
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        except Exception as e:
            device = torch.device("cuda")
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map["rgb"](imgs)
            # (N,B,D)
            feature = feature.reshape(-1, batch_size, *feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature, 0, 1)
            # (B,N*D)
            feature = feature.reshape(batch_size, -1)
            features.append(feature.to(device))
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if key == "point_cloud":
                    img = (
                        img[:, :, :3].permute(0, 2, 1)
                        if img.shape[2] == 6 or img.shape[2] == 3
                        else img
                    )
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key], (
                    f"{img.shape} vs {self.key_shape_map[key]}"
                )
                if hasattr(self, "preprocessor"):
                    img, depth = self.preprocessor(img)
                    feature = self.key_model_map[key](img, depth)
                else:
                    # print(f"{key}: {img.shape}")
                    img = self.key_transform_map[key](img)
                    feature = self.key_model_map[key](img)
                features.append(feature.to(device))
                # print(f"{key}: {features[-1].device}")
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data.to(device))
            # print(f"{key}: {features[-1].device}")

        # concatenate all features
        result = torch.cat(features, dim=-1)
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
