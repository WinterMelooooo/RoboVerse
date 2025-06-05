from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.model.vision.dp3_encoder import PointNetLayerEncoderXYZ
from torchvision.models import resnet18
import torch.nn as nn
import torch
import torch.nn.functional as F

class FusionLateEncoder(nn.Module):
    def __init__(self, rgb_params, pntcloud_params, action_params, fusion_params, add_uv=False):
        '''
        Args:
            rgb_params: parameters for the RGB encoder
            pntcloud_params: parameters for the point cloud encoder
                in_channels: number of input channels for point cloud
                pntcloud_output_channels: output dimension for point cloud
                pntcloud_layernorm: whether to use layer normalization in point cloud encoder
                pntcloud_final_norm: whether to apply final normalization in point cloud encoder
            fusion_params: parameters for the fusion layer
            add_uv: boolean indicating whether to add UV information
        '''
        super(FusionLateEncoder, self).__init__()
        self.rgb_params = rgb_params
        self.pntcloud_params = pntcloud_params
        self.action_params = action_params
        self.fusion_params = fusion_params
        self.add_uv = add_uv
        self.rgb_shape = fusion_params.rgb_shape_meta[1:]

        self.resnet_out_channels = [64, 128, 256, 512]
        self.resnet = resnet18(weights=None)
        self.resnet.avgpool = nn.Identity()
        self.resnet.fc = nn.Identity()
        self.resnet.mlp_layer2 = nn.Sequential(
            nn.Linear(self.resnet_out_channels[1], self.rgb_params.rgb_hidden_dim),
            nn.LayerNorm(self.rgb_params.rgb_hidden_dim),
            nn.ReLU(),
        )
        self.resnet.mlp_layer3 = nn.Sequential(
            nn.Linear(self.resnet_out_channels[2], self.rgb_params.rgb_hidden_dim),
            nn.LayerNorm(self.rgb_params.rgb_hidden_dim),
            nn.ReLU(),
        )
        self.resnet.mlp_layer4 = nn.Sequential(
            nn.Linear(self.resnet_out_channels[3], self.rgb_params.rgb_hidden_dim),
            nn.LayerNorm(self.rgb_params.rgb_hidden_dim),
            nn.ReLU(),
        )

        self.pointnet = PointNetLayerEncoderXYZ(
            in_channels=pntcloud_params.pntcloud_in_channels,
            out_channels=fusion_params.output_channels,
            use_layernorm=pntcloud_params.pntcloud_layernorm,
            final_norm=pntcloud_params.pntcloud_final_norm,
        )
        self.pointnet.pool = nn.Identity()
        self.pointnet.final_projection = nn.Identity()

        self.post_fusion = nn.Sequential(

        )

        if self.add_uv:
            self.uv_mlp = nn.Sequential(
                nn.Linear(2, fusion_params.uv_hidden_dim),
                nn.LayerNorm(fusion_params.uv_hidden_dim),
                nn.ReLU(),
            )

        if self.action_params.func == "identity":
            self.action_model = nn.Identity()
        elif self.action_params.func == "mlp":
            self.action_model = self.action_params.model


        if self.fusion_params.fusion_func == "cat":
            self.fusion_func = lambda *args: torch.cat([arg for arg in args if arg is not None], dim=-1)
        elif self.fusion_params.fusion_func == "mutual_attention":
            raise NotImplementedError("Mutual attention fusion is not implemented yet.")
        else:
            raise ValueError(f"Unknown fusion function: {self.fusion_params.fusion_func}")

        if self.fusion_params.pool == "maxpool":
            self.pool = lambda x: torch.max(x, 1)[0]
        else:
            raise ValueError(f"Unknown pooling function: {self.fusion_params.pool}")

        self.last_block_channel = 3* self.rgb_params.rgb_hidden_dim + self.pointnet.block_channel[-1] + self.fusion_params.uv_hidden_dim
        self.final_projection = nn.Sequential(
                nn.Linear(self.last_block_channel, self.fusion_params.output_channels),
                nn.LayerNorm(self.fusion_params.output_channels)
        )

    def forward(self, rgb, pntcloud, action):
        uv = pntcloud[..., -2:]
        pntcloud = pntcloud[..., :3]
        pointcloud_features = self.pointnet(pntcloud) # [B, N, 256]
        img_features = self._get_img_features(rgb, uv) # [B, N, 3*rgb_hidden_dim]
        action_features = self.action_model(action) # [B, N, action_dim]
        uv_features = None
        if self.add_uv:
            uv_features = self.uv_mlp(uv) # [B, N, uv_hidden_dim]
        fused = self.fusion_func(
            img_features, pointcloud_features, uv_features
        )
        fused = self.pool(fused)  # [B, action_dim + 3*rgb_hidden_dim + uv_hidden_dim]
        fused = self.final_projection(fused)  # [B, output_channels]
        fused = torch.cat([fused, action_features], dim=-1)  # [B, output_channels + action_dim]
        return fused

    def _get_img_features(self, rgb, uv, sample_mode="bilinear"):
        rgb = self.resnet.conv1(rgb)
        rgb = self.resnet.bn1(rgb)
        rgb = self.resnet.relu(rgb)
        rgb = self.resnet.maxpool(rgb)

        rgb = self.resnet.layer1(rgb)
        feat_map2 = self.resnet.layer2(rgb) #[B, 128, H/8, W/8]
        feat_map3 = self.resnet.layer3(feat_map2) # [B, 256, H/16, W/16]
        feat_map4 = self.resnet.layer4(feat_map3) # [B, 512, H/32, W/32]
        feat2 = self._sample_features_from_uv(feat_map2, uv, self.rgb_shape, sample_mode) # [B, N, 128]
        feat2 = self.resnet.mlp_layer2(feat2) # [B, N, rgb_hidden_dim]
        feat3 = self._sample_features_from_uv(feat_map3, uv, self.rgb_shape, sample_mode) # [B, N, 256]
        feat3 = self.resnet.mlp_layer3(feat3) # [B, N, rgb_hidden_dim]
        feat4 = self._sample_features_from_uv(feat_map4, uv, self.rgb_shape, sample_mode) # [B, N, 512]
        feat4 = self.resnet.mlp_layer4(feat4) # [B, N, rgb_hidden_dim]
        img_feats = torch.cat([feat2, feat3, feat4], dim=-1) # [B, N, 3*rgb_hidden_dim]
        return img_feats

    def _sample_features_from_uv(self,
                                 feat_map: torch.Tensor,
                                 uv: torch.Tensor,
                                 img_shape: list,
                                 sample_mode: str) -> torch.Tensor:
        """
        Args:
            feat_map (torch.Tensor): [B, C, Hc, Wc]。
            uv (torch.Tensor): [B, N, 2]
                uv[..., 0] is height direction, range in [0, img_h-1].
                uv[..., 1] is width direction, range in [0, img_w-1].
            img_h (int): height of the original RGB image.
            img_w (int): width of the original RGB image.

        Returns:
            torch.Tensor: [B, N, C]。
                ret[i, j, k] Stands for the value i-th batch, j-th point, and k-th channel.
        """
        img_h, img_w = img_shape

        x_norm = (uv[..., 1] / (img_w - 1)) * 2 - 1  # [-1,1]
        y_norm = (uv[..., 0] / (img_h - 1)) * 2 - 1  # [-1,1]
        grid = torch.stack((x_norm, y_norm), dim=-1)   # [B, N, 2]
        grid = grid.unsqueeze(2) # [B, N, 1, 2]
        sampled = F.grid_sample(feat_map,
                                grid,
                                mode=sample_mode,
                                padding_mode='border',
                                align_corners=True) # [B, C, N, 1]

        sampled = sampled.squeeze(-1)       # [B, C, N]
        sampled = sampled.permute(0, 2, 1)  # [B, N, C]

        return sampled
