import numpy as np
import torch
from torch import nn


class MultimodalBackbone(nn.Module):
    def __init__(
        self, rgb_backbone, pcd_backbone, proj_dim, fusion_func_name, num_heads
    ):
        super(MultimodalBackbone, self).__init__()
        self.rgb_backbone = rgb_backbone
        self.pcd_backbone = pcd_backbone
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.fusion_func = self._get_fusion_func(fusion_func_name)
        self.cross_atten = nn.MultiheadAttention(
            embed_dim=proj_dim, num_heads=num_heads, batch_first=True
        )
        self.img_proj = nn.Linear(rgb_backbone.num_channels, proj_dim)
        self.pcd_proj = nn.Linear(pcd_backbone.num_channels, proj_dim)
        self.modality_embed = nn.Embedding(3, proj_dim)

    def forward(self, obs):
        rgb_features, rgb_pos_enc = self.rgb_backbone(obs)
        pcd_features, pcd_pos_enc = self.pcd_backbone(obs)

        rgb_features = self._apply_positional_encoding(
            rgb_features, rgb_pos_enc
        )  # B, C, H, W
        rgb_features = rgb_features.flatten(2).transpose(1, 2)  # B, H*W, C
        rgb_features = self.img_proj(rgb_features)  # B, H*W, proj_dim
        rgb_features = self._apply_modality_embedding(
            rgb_features, modality=0
        )  # B, H*W, proj_dim

        pcd_features = self._apply_positional_encoding(
            pcd_features, pcd_pos_enc
        )  # B, D, 1, N
        pcd_features = pcd_features.flatten(2).transpose(1, 2)  # B, 1*N, D
        pcd_features = self.pcd_proj(pcd_features)  # B, 1*N, proj_dim
        pcd_features = self._apply_modality_embedding(
            pcd_features, modality=1
        )  # B, 1*N, proj_dim

        rgb_fused, _ = self.cross_atten(
            rgb_features, pcd_features, pcd_features
        )  # B, H*W, proj_dim
        rgb_fused += rgb_features
        pcd_fused, _ = self.cross_atten(
            pcd_features, rgb_features, rgb_features
        )  # B, 1*N, proj_dim
        pcd_fused += pcd_features

        fused = self.fusion_func(rgb_fused, pcd_fused)  # B, H*W+1*N, proj_dim
        fused = fused.transpose(1, 2).unsqueeze(2)  # B, proj_dim, 1, H*W+1*N

        fused_pos_enc = self._fuse_pos_enc(
            rgb_pos_enc, pcd_pos_enc
        )  # (B, C, 1, H*W + N)
        return fused, fused_pos_enc

    def _apply_modality_embedding(self, features, modality):
        return features + self.modality_embed(
            torch.tensor(modality, device=features.device, dtype=torch.long)
        )

    def _apply_positional_encoding(self, features, pos_enc):
        B, _, _, _ = features.shape
        pos_enc = pos_enc.expand(B, -1, -1, -1)
        return features + pos_enc

    def _fuse_pos_enc(
        self,
        rgb_pos_enc,  # (B, C, H, W)
        pcd_pos_enc,  # (B, C, 1, N)
    ):
        """
        把图像和点云的 pos_enc 在空间维度上拼成一个融合的 pos_enc。

        返回：Tensor of shape (B, C, 1, H*W + N)
        """
        B, C, H, W = rgb_pos_enc.shape
        _, _, _, N = pcd_pos_enc.shape

        # 1) 把图像的 (H, W) flatten 成一个维度
        rgb_flat = rgb_pos_enc.view(B, C, 1, H * W)  # (B, C, 1, H*W)

        # 2) 确保点云也是 (B, C, 1, N)，直接拼接
        #    pcd_pos_enc 本身就是 (B, C, 1, N)

        # 3) 在最末尾那一维拼起来
        fused = torch.cat([rgb_flat, pcd_pos_enc], dim=3)  # (B, C, 1, H*W + N)
        return fused

    def _get_fusion_func(self, fusion_func_name):
        if fusion_func_name == "concat":
            return lambda x, y: torch.cat([x, y], dim=1)
        elif fusion_func_name == "self_atten":
            self.self_atten = nn.MultiheadAttention(
                embed_dim=self.proj_dim, num_heads=self.num_heads, batch_first=True
            )

            def fusion_func(img, pcd):
                fused = torch.cat([img, pcd], dim=1)  # B, H*W+1*N, proj_dim
                fused, _ = self.self_atten(fused, fused, fused)  # B, H*W+1*N, proj_dim
                return fused

            return fusion_func
        else:
            raise ValueError(f"Unknown fusion function: {fusion_func_name}")
