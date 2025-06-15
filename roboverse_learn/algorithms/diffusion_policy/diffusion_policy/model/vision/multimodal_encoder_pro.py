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
        self.fusion_func = self._get_fusion_func(fusion_args)
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


    def forward(self, obs_dict, use_gt_sensor):
        for key in self.sensor_state_keys:
            if key not in obs_dict:
                raise ValueError(f"Key {key} not found in obs_dict. Available keys: {obs_dict.keys()}")
        batch_size = None
        batch_size_pnt = None
        img_features = list()
        low_dim_features = list()
        pntcloud_features = list()
        sensor_state_features = list()
        device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
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

        for key in self.sensor_state_keys:
            sensor_state = obs_dict[key]
            if batch_size is None:
                batch_size = sensor_state.shape[0]
            else:
                assert batch_size == sensor_state.shape[0]
            assert sensor_state.shape[1:] == self.key_shape_map[key]
            sensor_state = self.key_transform_map[key](sensor_state)
            feature = self.key_model_map[key](sensor_state) #[B, D]
            sensor_state_features.append(feature.to(device)) #[N, B, D]
            # print(f"{key}: {features[-1].device}")

        # Dropout
        if torch.rand((), device=device) < self.dropout:
            if torch.rand((), device=device) < 0.5:
                img_features = [f * 0.0 for f in img_features]
            else:
                pntcloud_features = [f * 0.0 for f in pntcloud_features]

        # concatenate all features
        fused = self.fusion_func(img_features=img_features, low_dim_features=low_dim_features, pntcloud_features=pntcloud_features, sensor_state_features=sensor_state_features)
        predict_sensors_state = self.prediction_model(fused) # Sensor_Name: [B, D]

        target_sensors_state = {
            key: obs_dict[key] for key in self.sensor_state_keys
        } if use_gt_sensor else predict_sensors_state

        for sensor_name, predict_sensor_state in target_sensors_state.items():
            if predict_sensor_state.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: {predict_sensor_state.shape} vs {batch_size}"
                )
            predict_sensor_state = self.key_transform_map[sensor_name](predict_sensor_state)
            pred_sensor_feature = self.key_model_map[sensor_name](predict_sensor_state) #[B, D]
            sensor_state_features.append(pred_sensor_feature.to(device))

        fused = self.fusion_func(img_features=img_features, low_dim_features=low_dim_features, pntcloud_features=pntcloud_features, sensor_state_features=sensor_state_features)
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
        if fusion_method == "cat":
            return self._cat_features
        elif fusion_method == "cross_attention":
            sig = inspect.signature(self._get_attention_func)
            all_params = list(sig.parameters.keys())   # ['self', 'embed_dim', 'num_heads', ...]
            kwargs = {k: fusion_args[k] for k in all_params if k in fusion_args}
            return self._get_attention_func(**kwargs)
        elif fusion_method == "joint_attention":
            sig = inspect.signature(self._get_joint_attention_func)
            all_params = list(sig.parameters.keys())   # ['self', 'embed_dim', 'num_heads', ...]
            kwargs = {k: fusion_args[k] for k in all_params if k in fusion_args}
            return self._get_joint_attention_func(**kwargs)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

    def _get_attention_func(self, embed_dim: int,
                             num_heads: int,
                             img_dim: int,
                             pc_dim: int,
                             state_dim: int,
                             sensor_dim: int,
                             mutual_attention: bool = True,
                             post_fusion_func: str = "sum",
                             use_residual: bool = False,
                             use_independent_attention: bool = False,
                             use_modality_encoding: bool = False):
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
            self.use_independent_attention = use_independent_attention
            self.use_modality_encoding = use_modality_encoding
            if use_independent_attention:
                self.cross_attn_key = nn.MultiheadAttention(embed_dim, num_heads)
            if use_modality_encoding:
                self.modality_embed = nn.Embedding(3+2*len(self.sensor_state_keys), embed_dim)  # 3+n modalities: img, pc, state, n sensor(present + predict)

            self.use_residual = use_residual
            cprint(f"[Cross Attention]: use residual: {use_residual}", "cyan")
            cprint(f"[Cross Attention]: mutual_attention: {mutual_attention}", "cyan")
            cprint(f"[Cross Attention]: post_fusion_func: {post_fusion_func}", "cyan")
            cprint(f"[Cross Attention]: use_independent_attention: {use_independent_attention}", "cyan")
            cprint(f"[Cross Attention]: use_modality_encoding: {use_modality_encoding}", "cyan")
            self.img_proj = nn.Sequential(nn.Linear(img_dim, embed_dim))
            self.pc_proj = nn.Sequential(nn.Linear(pc_dim, embed_dim))
            self.state_proj = nn.Sequential(nn.Linear(state_dim, embed_dim))
            self.sensor_proj = nn.Sequential(nn.Linear(sensor_dim, embed_dim))
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

    def _cross_attention_features(self, img_features, low_dim_features, pntcloud_features, sensor_state_features):
        img_feats = [self.img_proj(f) for f in img_features] # [N1, B, embed_dim]
        pc_feats = [self.pc_proj(f) for f in pntcloud_features] # [N2, B, embed_dim]
        low_dim_feats = [self.state_proj(f) for f in low_dim_features] # [N3, B, embed_dim]
        sensor_state_feats = [self.sensor_proj(f) for f in sensor_state_features] # [N4, B, embed_dim]
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
        if self.fusion_args.get("main_feat", "img") == "img":
            main_feat = img_feats # [N1+N4, B, embed_dim]
            key_feats = pc_feats + low_dim_feats + sensor_state_feats # [N2+N3, B, embed_dim]
        elif self.fusion_args.get("main_feat", "img") == "pcd":
            main_feat = pc_feats # [N2+N4, B, embed_dim]
            key_feats = img_feats + low_dim_feats + sensor_state_feats # [N1+N3, B, embed_dim]
        else:
            raise ValueError(f"Unknown main_feat: {self.fusion_args.get('main_feat', 'img')}")
        main_feats = torch.stack(main_feat, dim=0) # [N1+N4, B, embed_dim]
        key_feats = torch.stack(key_feats, dim=0) # [N2+N3, B, embed_dim]
        attn_output, attn_map = self.cross_attn(main_feats, key_feats, key_feats)
        if self.use_residual:
            attn_output = attn_output + main_feats
        fused = attn_output.mean(dim=0)
        if self.mutual_attention:
            if self.use_independent_attention:
                attn_output, attn_map = self.cross_attn_key(key_feats, main_feats, main_feats)
            else:
                attn_output, attn_map = self.cross_attn(key_feats, main_feats, main_feats)

            if self.use_residual:
                attn_output = attn_output + key_feats
            if self.post_fusion_func == "sum":
                fused += attn_output.mean(dim=0)
            elif self.post_fusion_func == "mlp":
                fused = torch.cat([fused, attn_output.mean(dim=0)], dim=-1)
                fused = self.post_fusion_mlp(fused)
            elif self.post_fusion_func == "cat":
                fused = torch.cat([fused, attn_output.mean(dim=0)], dim=-1)
            else:
                raise ValueError(f"Unknown post_fusion_func: {self.post_fusion_func}")
        return fused

    def _get_joint_attention_func(self,
                             embed_dim: int,
                             num_heads: int,
                             img_dim: int,
                             pc_dim: int,
                             state_dim: int,
                             sensor_dim: int,
                             pooling_func: str = "mean",
                             use_residual: bool = False,
                             use_modality_encoding: bool = False):
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads)
            self.use_modality_encoding = use_modality_encoding
            self.pooling_func = pooling_func
            self.use_residual = use_residual
            cprint(f"[Joint Attention]: use residual: {use_residual}", "cyan")
            cprint(f"[Joint Attention]: use_modality_encoding: {use_modality_encoding}", "cyan")
            cprint(f"[Joint Attention]: pooling_func: {pooling_func}", "cyan")
            if use_modality_encoding:
                self.modality_embed = nn.Embedding(3+2*len(self.sensor_state_keys), embed_dim)  # 3+n modalities: img, pc, state, n sensor(present + predict)
            self.img_proj = nn.Sequential(nn.Linear(img_dim, embed_dim))
            self.pc_proj = nn.Sequential(nn.Linear(pc_dim, embed_dim))
            self.state_proj = nn.Sequential(nn.Linear(state_dim, embed_dim))
            self.sensor_proj = nn.Sequential(nn.Linear(sensor_dim, embed_dim))

            return self._joint_attention_features

    def _joint_attention_features(self, img_features, low_dim_features, pntcloud_features, sensor_state_features):
        img_feats = [self.img_proj(f) for f in img_features] # [N1, B, embed_dim]
        pc_feats = [self.pc_proj(f) for f in pntcloud_features] # [N2, B, embed_dim]
        low_dim_feats = [self.state_proj(f) for f in low_dim_features] # [N3, B, embed_dim]
        sensor_state_feats = [self.sensor_proj(f) for f in sensor_state_features] # [N4, B, embed_dim]
        if self.use_modality_encoding:
            device = img_feats[0].device
            embed = lambda idx: self.modality_embed(torch.tensor(idx, device=device, dtype=torch.long))  # (embed_dim,)
            n_feats = [len(img_feats), len(pc_feats), len(low_dim_feats), len(sensor_state_feats)]
            start_img, start_pc, start_state, start_sensor = 0, n_feats[0], n_feats[0] + n_feats[1], n_feats[0] + n_feats[1] + n_feats[2]

            img_feats = [f + embed(idx+start_img) for idx, f in enumerate(img_feats)]
            pc_feats = [f + embed(idx+start_pc) for idx, f in enumerate(pc_feats)]
            low_dim_feats = [f + embed(idx+start_state) for idx, f in enumerate(low_dim_feats)]
            sensor_state_feats = [f + embed(idx+start_sensor) for idx, f in enumerate(sensor_state_feats)]
        feats = img_feats + pc_feats + low_dim_feats + sensor_state_feats
        feats = torch.stack(feats, dim=0) # [N1+N2+N3+N4, B, embed_dim]
        attn_output, _ = self.cross_attn(feats, feats, feats)
        if self.use_residual:
            attn_output = attn_output + feats
        if self.pooling_func == "mean":
            fused = attn_output.mean(dim=0)
        else:
            raise ValueError(f"Unknown pooling function: {self.pooling_func}")

        return fused

    def _cat_features(self, *args):
        """
        Concatenate all features along the last dimension.
        """
        all_features = list(args)
        return torch.cat(all_features, dim=-1)
