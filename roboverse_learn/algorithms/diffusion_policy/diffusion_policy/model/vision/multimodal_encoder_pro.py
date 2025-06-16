import copy
import os
from typing import Dict, Tuple, Union
import inspect
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from roboverse_learn.algorithms.utils.channelwise_dropout import ChannelWiseDropout
from termcolor import cprint

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




class MultiModalEncoderPro(ModuleAttrMixin):
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

            if type == "rgb" or type == "rgbd":
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
                    if img_encoder_args.get("pretrained", None):
                        cprint(f"Loading pretrained weights for {key} from {img_encoder_args.pretrained}", "cyan")
                        ckpt = torch.load(img_encoder_args.pretrained, map_location="cpu")
                        full_sd = ckpt["state_dicts"]["model"]
                        prefix = "obs_encoder.key_model_map.head_cam"
                        sub_sd = {
                            k[len(prefix) + 1:]: v
                            for k, v in full_sd.items()
                            if k.startswith(prefix + ".")
                        }
                        this_model.load_state_dict(sub_sd, strict=True)

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
        self.fusion_args = fusion_args
        self.dropout = fusion_args.get("dropout", 0.0)
        self.dropout_model = nn.Dropout(p=self.dropout) if not fusion_args.get("use_channel_wise_dropout", False) else ChannelWiseDropout(p=self.dropout)
        cprint(f"[MultiModal Encoder]: channel wise dropout: {fusion_args.get('use_channel_wise_dropout', False)}", "cyan")
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
        extra = None
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
            feature = self.key_model_map[key](pnt_cloud[..., :3])  # assuming the first 3 dimensions are x, y, z
            pntcloud_features.append(feature.to(device))
            if pnt_cloud.shape[-1] > 3:
                extra = pnt_cloud[..., 3:]  # assuming the extra features are in the last dimensions
            # print(f"{key}: {features[-1].device}")

        # Dropout
        if self.training and torch.rand((), device=device) < self.dropout:
            if torch.rand((), device=device) < 0.5:
                img_features = [f * 0.0 for f in img_features]
            else:
                pntcloud_features = [f * 0.0 for f in pntcloud_features]

        # concatenate all features
        fused = self.fusion_func(img_features=img_features, low_dim_features=low_dim_features, pntcloud_features=pntcloud_features, extra=extra)
        fused = self.dropout_model(fused)
        return fused

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
            elif type == "rgbd":
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
        if  fusion_method == "cross_attention":
            sig = inspect.signature(self._get_attention_func)
            all_params = list(sig.parameters.keys())   # ['self', 'embed_dim', 'num_heads', ...]
            kwargs = {k: fusion_args[k] for k in all_params if k in fusion_args}
            return self._get_attention_func(**kwargs)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    def _get_attention_func(self, embed_dim: int,
                             num_heads: int,
                             img_dim: int,
                             pc_dim: int,
                             state_dim: int,
                             mutual_attention: bool = True,
                             post_fusion_func: str = "sum",
                             norm_proj: bool = False,
                             use_residual: bool = False,
                             use_independent_attention: bool = False,
                             use_modality_encoding: bool = False):
            self.cross_attn_rgb_pnt = nn.MultiheadAttention(embed_dim, num_heads)
            self.cross_attn_vis_state = nn.MultiheadAttention(embed_dim, num_heads)
            self.use_independent_attenion = use_independent_attention
            self.use_modality_encoding = use_modality_encoding
            if use_independent_attention:
                self.cross_attn_pnt_rgb = nn.MultiheadAttention(embed_dim, num_heads)
                self.cross_attn_state_vis = nn.MultiheadAttention(embed_dim, num_heads)
            if use_modality_encoding:
                self.modality_embed = nn.Embedding(3, embed_dim)  # 3 modalities: img, pc, state

            self.use_residual = use_residual
            cprint(f"[Cross Attention]: use residual: {use_residual}", "cyan")
            cprint(f"[Cross Attention]: mutual_attention: {mutual_attention}", "cyan")
            cprint(f"[Cross Attention]: post_fusion_func: {post_fusion_func}", "cyan")
            cprint(f"[Cross Attention]: use_independent_attention: {use_independent_attention}", "cyan")
            cprint(f"[Cross Attention]: use_modality_encoding: {use_modality_encoding}", "cyan")
            if norm_proj:
                num_groups = embed_dim // 16 if embed_dim % 16 == 0 else embed_dim // 8
                self.img_norm_layer = nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=embed_dim,
                )
                self.pc_norm_layer = nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=embed_dim,
                )
                self.state_norm_layer = nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=embed_dim,
                )
            else:
                self.img_norm_layer = nn.Identity()
                self.pc_norm_layer = nn.Identity()
                self.state_norm_layer = nn.Identity()
            self.img_proj = nn.Sequential(nn.Linear(img_dim, embed_dim), self.img_norm_layer)
            self.pc_proj = nn.Sequential(nn.Linear(pc_dim, embed_dim), self.pc_norm_layer)
            self.state_proj = nn.Sequential(nn.Linear(state_dim, embed_dim), self.state_norm_layer)
            self.mutual_attention = mutual_attention
            self.post_fusion_func = post_fusion_func
            if post_fusion_func == "sum":
                pass
            elif post_fusion_func == "mlp":
                self.post_fusion_mlp = nn.Sequential(
                    nn.Linear(2*embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, embed_dim)
                )
            elif post_fusion_func == "cat":
                pass
            else:
                raise ValueError(f"Unknown post_fusion_func: {post_fusion_func}")
            return self._cross_attention_features

    def _cross_attention_features(self, img_features, low_dim_features, pntcloud_features, extra=None):
        img_feats = [self.img_proj(f) for f in img_features] # [N1, B, embed_dim]
        pc_feats = [self.pc_proj(f) for f in pntcloud_features] # [N2, B, embed_dim]
        low_dim_feats = [self.state_proj(f) for f in low_dim_features] # [N3, B, embed_dim]
        if self.use_modality_encoding:
            device = img_feats[0].device
            e_img  = self.modality_embed(torch.tensor(0, device=device, dtype=torch.long))  # (embed_dim,)
            e_pc   = self.modality_embed(torch.tensor(1, device=device, dtype=torch.long))
            e_state= self.modality_embed(torch.tensor(2, device=device, dtype=torch.long))
            img_feats = [f + e_img for f in img_feats]
            pc_feats = [f + e_pc for f in pc_feats]
            low_dim_feats = [f + e_state for f in low_dim_feats]
        img_feats = torch.stack(img_feats, dim=0)  # [N1, B, embed_dim]
        pc_feats = torch.stack(pc_feats, dim=0)
        state_feats = torch.stack(low_dim_feats, dim=0)
        rgb_2_pc_attn_output, _ = self.cross_attn_rgb_pnt(img_feats, pc_feats, pc_feats)
        if self.use_residual:
            rgb_2_pc_attn_output = rgb_2_pc_attn_output + img_feats
        vis_fused = rgb_2_pc_attn_output.mean(dim=0)
        if self.mutual_attention:
            if self.use_independent_attenion:
                pc_2_rgb_attn_output, attn_map = self.cross_attn_pnt_rgb(pc_feats, img_feats, img_feats)
            else:
                pc_2_rgb_attn_output, attn_map = self.cross_attn_rgb_pnt(pc_feats, img_feats, img_feats)

            if self.use_residual:
                pc_2_rgb_attn_output = pc_2_rgb_attn_output + pc_feats
            if self.post_fusion_func == "sum":
                vis_fused += pc_2_rgb_attn_output.mean(dim=0)
            elif self.post_fusion_func == "mlp":
                vis_fused = torch.cat([vis_fused, pc_2_rgb_attn_output.mean(dim=0)], dim=-1)
                vis_fused = self.post_fusion_mlp(vis_fused)
            elif self.post_fusion_func == "cat":
                vis_fused = torch.cat([vis_fused, pc_2_rgb_attn_output.mean(dim=0)], dim=-1)
            else:
                raise ValueError(f"Unknown post_fusion_func: {self.post_fusion_func}")
        vis_2_state_attn_output, _ = self.cross_attn_vis_state(vis_fused.unsqueeze(0), state_feats, state_feats)
        if self.use_residual:
            vis_2_state_attn_output = vis_2_state_attn_output + vis_fused.unsqueeze(0)
        fused = vis_2_state_attn_output.mean(dim=0)
        if self.mutual_attention:
            if self.use_independent_attenion:
                state_2_vis_attn_output, _ = self.cross_attn_state_vis(state_feats, vis_fused.unsqueeze(0), vis_fused.unsqueeze(0))
            else:
                state_2_vis_attn_output, _ = self.cross_attn_vis_state(state_feats, vis_fused.unsqueeze(0), vis_fused.unsqueeze(0))

            if self.use_residual:
                state_2_vis_attn_output = state_2_vis_attn_output + state_feats
            if self.post_fusion_func == "sum":
                fused += state_2_vis_attn_output.mean(dim=0)
            elif self.post_fusion_func == "mlp":
                fused = torch.cat([fused, state_2_vis_attn_output.mean(dim=0)], dim=-1)
                fused = self.post_fusion_mlp(fused)
            elif self.post_fusion_func == "cat":
                fused = torch.cat([fused, state_2_vis_attn_output.mean(dim=0)], dim=-1)
            else:
                raise ValueError(f"Unknown post_fusion_func: {self.post_fusion_func}")

        return fused
