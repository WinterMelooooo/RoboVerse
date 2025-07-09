from einops.layers.torch import Rearrange
from torch import nn


def get_resnet(lr_backbone, masks, backbone_name, dilation, positional_encoding_args):
    from .backbone import Backbone, Joiner, build_position_encoding

    position_embedding = build_position_encoding(positional_encoding_args)
    train_backbone = lr_backbone > 0
    return_interm_layers = masks
    backbone = Backbone(backbone_name, train_backbone, return_interm_layers, dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def get_dp3(hidden_dim, **encoder_args):
    from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.model.vision.dp3_encoder import (
        PointNetEncoderXYZ,
    )

    from .backbone import PcdJoiner
    from .position_encoding import BatchedPcdPositionEmbedding

    encoder = PointNetEncoderXYZ(**encoder_args)
    encoder.pool = nn.Identity()
    encoder.final_projection = nn.Sequential(encoder.final_projection, Rearrange("b n d -> b d 1 n"))
    position_embedding = BatchedPcdPositionEmbedding(hidden_dim=hidden_dim)
    model = PcdJoiner(encoder, position_embedding)
    model.num_channels = encoder.out_channels
    return model


def get_spUnet(hidden_dim, **encoder_args):
    from roboverse_learn.algorithms.diffusion_policy.diffusion_policy.model.vision.spUnet import SpUnetACTEncoder

    from .backbone import VoxelPcdJoiner
    from .position_encoding import BatchedPcdPositionEmbedding

    encoder = SpUnetACTEncoder(**encoder_args)
    position_embedding = BatchedPcdPositionEmbedding(hidden_dim=hidden_dim)
    model = VoxelPcdJoiner(encoder, position_embedding)
    model.num_channels = 96
    return model


def get_multimodal_backbone(rgb_backbone, pcd_backbone, proj_dim, fusion_func_name, num_heads, hidden_dim):
    from .backbone import MultiModalJoiner
    from .multimodal_backbone import MultimodalBackbone
    from .position_encoding import BatchedPcdPositionEmbedding
    encoder = MultimodalBackbone(rgb_backbone, pcd_backbone, proj_dim, fusion_func_name, num_heads=num_heads)
    encoder.num_channels = proj_dim
    return encoder
