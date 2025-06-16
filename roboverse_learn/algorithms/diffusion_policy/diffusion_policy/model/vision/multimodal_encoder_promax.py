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




class MultiModalEncoderProMax(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        img_model: Union[nn.Module, Dict[str, nn.Module]],
        pntcloud_model: Union[nn.Module, Dict[str, nn.Module]],
        robot_state_model: Union[nn.Module, Dict[str, nn.Module]],
        sensor_model: Union[nn.Module, Dict[str, nn.Module]] = None,
        prediction_model: nn.Module = None,
        img_encoder_args:ImgEncoderArgs = None,
        pntcloud_encoder_args:PntCloudEncoderArgs = None,
        robot_state_encoder_args:RobotStateEncoderArgs = None,
        sensor_state_encoder_args= None,
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
        sensor_state_keys = list()
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
            elif type == "force":
                sensor_state_keys.append(key)
                if sensor_state_encoder_args.get("share_model", True):
                    if isinstance(sensor_model, dict):
                        this_model = sensor_model[key]
                    else:
                        this_model = sensor_model
                else:
                    if isinstance(sensor_model, dict):
                        this_model = copy.deepcopy(sensor_model[key])
                    else:
                        this_model = copy.deepcopy(sensor_model)
                key_model_map[key] = this_model
                key_transform_map[key] = nn.Identity()
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        img_keys = sorted(img_keys)
        low_dim_keys = sorted(low_dim_keys)
        point_cloud_keys = sorted(point_cloud_keys)
        sensor_state_keys = sorted(sensor_state_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.img_keys = img_keys
        self.low_dim_keys = low_dim_keys
        self.point_cloud_keys = point_cloud_keys
        self.sensor_state_keys = sensor_state_keys
        self.key_shape_map = key_shape_map
        self.primary_fusion_func, self.secondary_fusion_func = self._get_fusion_func(fusion_args)
        self.fusion_args = fusion_args
        self.dropout = fusion_args.get("dropout", 0.0)
        self.dropout_model = nn.Dropout(p=self.dropout) if not fusion_args.get("use_channel_wise_dropout", False) else ChannelWiseDropout(p=self.dropout)
        self.prediction_model = prediction_model
        cprint(f"[MultiModal Encoder]: channel wise dropout: {fusion_args.get('use_channel_wise_dropout', False)}", "cyan")
        try:
            device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
        except Exception as e:
            device = torch.device("cuda")
        self.to(device)
        self.cuda_device = device

    def forward(self, obs_dict, use_gt_sensor):
        for key in self.sensor_state_keys:
            if not use_gt_sensor and key.endswith("_pred"):
                continue
            if key not in obs_dict:
                raise ValueError(f"Key {key} not found in obs_dict. Available keys: {obs_dict.keys()}")
        batch_size = None
        batch_size_pnt = None
        img_features = list()
        low_dim_features = list()
        pntcloud_features = list()
        pres_sensor_state_features = list()
        device = self.cuda_device
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

        for key in self.sensor_state_keys:
            if key.endswith("_pres"):
                pres_sensor_state = obs_dict[key] # [B, D0]
                if batch_size is None:
                    batch_size = pres_sensor_state.shape[0]
                else:
                    assert batch_size == pres_sensor_state.shape[0]
                assert pres_sensor_state.shape[1:] == self.key_shape_map[key]
                pres_sensor_state = self.key_transform_map[key](pres_sensor_state) #[B, D0]
                feature = self.key_model_map[key](pres_sensor_state) #[B, D]
                pres_sensor_state_features.append(feature.to(device)) #[N, B, D]
                # print(f"{key}: {features[-1].device}")

        # Dropout
        if torch.rand((), device=device) < self.dropout:
            if torch.rand((), device=device) < 0.5:
                img_features = [f * 0.0 for f in img_features]
            else:
                pntcloud_features = [f * 0.0 for f in pntcloud_features]

        # concatenate all features
        fused = self.primary_fusion_func(img_features=img_features, low_dim_features=low_dim_features, pntcloud_features=pntcloud_features, pres_sensor_state_features=pres_sensor_state_features)
        predict_sensors_state = self.prediction_model(fused) # Sensor_Name: [B, D]

        pred_sensors_state = {
            key: obs_dict[key] for key in self.sensor_state_keys if key.endswith("_pred")
        } if use_gt_sensor else predict_sensors_state

        pred_sensor_state_features = list()
        for sensor_name, predict_sensor_state in pred_sensors_state.items():
            if predict_sensor_state.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: {predict_sensor_state.shape} vs {batch_size}"
                )
            predict_sensor_state = self.key_transform_map[sensor_name](predict_sensor_state)
            pred_sensor_feature = self.key_model_map[sensor_name](predict_sensor_state) #[B, D]
            pred_sensor_state_features.append(pred_sensor_feature.to(device))

        fused = self.secondary_fusion_func(fused_features=fused, pred_sensor_state_features=pred_sensor_state_features)
        fused = self.dropout_model(fused)
        return fused, predict_sensors_state

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
            example_output, pred_sensor_state = self.forward(example_obs_dict, True)
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
                h, w = args.resize_shape[key]
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
        if fusion_method == "cross_attention":
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
                             sensor_dim: int,
                             name_secondary_fusion_func: str = "mean_pool_atten",
                             use_residual: bool = False,
                             use_independent_attention: bool = False,
                             use_modality_encoding: bool = False):
            self.cross_attn_rgb_pcd = nn.MultiheadAttention(embed_dim, num_heads)
            self.cross_attn_vis_state = nn.MultiheadAttention(embed_dim, num_heads)

            self.use_independent_attention = use_independent_attention
            self.use_modality_encoding = use_modality_encoding
            self.n_tokens = len(self.img_keys) + len(self.point_cloud_keys) + len(self.low_dim_keys) + len(self.sensor_state_keys)
            if use_independent_attention:
                self.cross_attn_pcd_rgb = nn.MultiheadAttention(embed_dim, num_heads)
                self.cross_attn_state_vis = nn.MultiheadAttention(embed_dim, num_heads)
            if use_modality_encoding:
                self.modality_embed = nn.Embedding(self.n_tokens, embed_dim)  # 3+n+1 modalities: img, pc, state, n sensor(present + predict), fused

            self.use_residual = use_residual
            self.name_secondary_fusion_func = name_secondary_fusion_func
            cprint(f"[Cross Attention]: use residual: {use_residual}", "cyan")
            cprint(f"[Cross Attention]: secondary_fusion_func: {name_secondary_fusion_func}", "cyan")
            cprint(f"[Cross Attention]: use_independent_attention: {use_independent_attention}", "cyan")
            cprint(f"[Cross Attention]: use_modality_encoding: {use_modality_encoding}", "cyan")
            self.img_proj = nn.Sequential(nn.Linear(img_dim, embed_dim))
            self.pc_proj = nn.Sequential(nn.Linear(pc_dim, embed_dim))
            self.state_proj = nn.Sequential(nn.Linear(state_dim, embed_dim))
            self.sensor_proj = nn.Sequential(nn.Linear(sensor_dim, embed_dim))

            if self.name_secondary_fusion_func == "cls_atten":
                self.cross_attn_cls_fuse = nn.MultiheadAttention(embed_dim, num_heads)
                self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

            elif self.name_secondary_fusion_func == "mean_pool_atten":
                from .models import AttentionPoolingHead
                self.pool =  AttentionPoolingHead(num_tokens=self.n_tokens)
                self.fusion_atten_pred = nn.MultiheadAttention(embed_dim, num_heads)
                if self.use_independent_attention:
                    self.pred_atten_fusion = nn.MultiheadAttention(embed_dim, num_heads)
            else:
                raise ValueError(f"Unknown secondary fusion function: {name_secondary_fusion_func}")
            return self._primary_cross_attention_features, self._secondary_cross_attention_features

    def _primary_cross_attention_features(self, img_features, low_dim_features, pntcloud_features, pres_sensor_state_features):
        img_feats = [self.img_proj(f) for f in img_features] # [N1, B, embed_dim]
        pc_feats = [self.pc_proj(f) for f in pntcloud_features] # [N2, B, embed_dim]
        low_dim_feats = [self.state_proj(f) for f in low_dim_features] # [N3, B, embed_dim]
        sensor_state_feats = [self.sensor_proj(f) for f in pres_sensor_state_features] # [N4, B, embed_dim]
        if self.use_modality_encoding:
            device = img_feats[0].device
            embed = lambda idx: self.modality_embed(torch.tensor(idx, device=device, dtype=torch.long))  # (embed_dim,)
            n_feats = [len(img_feats), len(pc_feats), len(low_dim_feats), len(sensor_state_feats)]
            start_img, start_pc, start_state, start_sensor = 0, n_feats[0], n_feats[0] + n_feats[1], n_feats[0] + n_feats[1] + n_feats[2]
            # 0, 1, 2, 3
            img_feats = [f + embed(idx+start_img) for idx, f in enumerate(img_feats)]
            pc_feats = [f + embed(idx+start_pc) for idx, f in enumerate(pc_feats)]
            low_dim_feats = [f + embed(idx+start_state) for idx, f in enumerate(low_dim_feats)]
            sensor_state_feats = [f + embed(idx+start_sensor) for idx, f in enumerate(sensor_state_feats)]
        img_feats = torch.stack(img_feats, dim=0) # [N1, B, embed_dim]
        pc_feats = torch.stack(pc_feats, dim=0) # [N2, B, embed_dim]
        state_feats = torch.stack(low_dim_feats+sensor_state_feats, dim=0) # [N3+N4, B, embed_dim]
        rgb_attn_output, attn_map = self.cross_attn_rgb_pcd(img_feats, pc_feats, pc_feats)
        if self.use_independent_attention:
            pcd_attn_output, attn_map = self.cross_attn_pcd_rgb(pc_feats, img_feats, img_feats)
        else:
            pcd_attn_output, attn_map = self.cross_attn_rgb_pcd(pc_feats, img_feats, img_feats)

        if self.use_residual:
            rgb_attn_output = rgb_attn_output + img_feats
            pcd_attn_output = pcd_attn_output + pc_feats

        vis_fused = torch.cat([rgb_attn_output, pcd_attn_output], dim=0)  # [N1+N2, B, embed_dim]

        vis_attn_output, attn_map = self.cross_attn_vis_state(vis_fused, state_feats, state_feats)
        if self.use_independent_attention:
            state_attn_output, attn_map = self.cross_attn_state_vis(state_feats, vis_fused, vis_fused)
        else:
            state_attn_output, attn_map = self.cross_attn_vis_state(state_feats, vis_fused, vis_fused)
        if self.use_residual:
            vis_attn_output = vis_attn_output + vis_fused
            state_attn_output = state_attn_output + state_feats

        # fuse attention outputs
        feats = torch.cat([vis_attn_output, state_attn_output], dim=0)  # [N1+N2+N3+N4, B, embed_dim]
        return feats

    def _secondary_cross_attention_features(self, fused_features, pred_sensor_state_features):
        pred_sensor_state_features = [self.sensor_proj(f) for f in pred_sensor_state_features] # [N4, B, embed_dim]
        if self.use_modality_encoding:
            device = fused_features[0].device
            embed = lambda idx: self.modality_embed(torch.tensor(idx, device=device, dtype=torch.long))
            fused_start_idx = 0
            sensor_start_idx = len(fused_features)
            fused_features = [f + embed(idx+fused_start_idx) for idx, f in enumerate(fused_features)]
            pred_sensor_state_features = [f + embed(idx+sensor_start_idx) for idx, f in enumerate(pred_sensor_state_features)]
        fused_feats = torch.stack(fused_features, dim=0) # [N1+N2+N3+N4, B, embed_dim]
        pred_sensor_feats = torch.stack(pred_sensor_state_features, dim=0) # [N4, B, embed_dim]
        if self.name_secondary_fusion_func == "cls_atten":
            fused = torch.cat([fused_feats, pred_sensor_feats], dim=0)  # [N1+N2+N3+N4+N4, B, embed_dim]
            B, D = fused.shape[-2:]
            cls = self.cls_token.expand(1, B, D)
            feat, _ = self.cross_attn_cls_fuse(cls, fused, fused) # [1, B, embed_dim]
            feat = feat.squeeze(0)  # [B, embed_dim]
            return feat

        elif self.name_secondary_fusion_func == "mean_pool_atten":
            fused_attn_output, attn_map = self.fusion_atten_pred(fused_feats, pred_sensor_feats, pred_sensor_feats)
            if self.use_independent_attention:
                state_attn_output, attn_map = self.pred_atten_fusion(pred_sensor_feats, fused_feats, fused_feats)
            else:
                state_attn_output, attn_map = self.fusion_atten_pred(pred_sensor_feats, fused_feats, fused_feats)
            if self.use_residual:
                fused_attn_output = fused_attn_output + fused_feats
                state_attn_output = state_attn_output + pred_sensor_feats
            feat = torch.cat([fused_attn_output, state_attn_output], dim=0)  # [N1+N2+N3+N4+N4, B, embed_dim]
            feat = self.pool(feat)  # [B, embed_dim]
            return feat
        else:
            raise ValueError(f"Unknown secondary fusion function: {self.secondary_fusion_func}")
