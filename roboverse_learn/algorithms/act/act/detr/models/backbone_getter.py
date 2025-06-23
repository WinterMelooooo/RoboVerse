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
    from .position_encoding import PositionEmbeddingPcd

    encoder = PointNetEncoderXYZ(**encoder_args)
    encoder.pool = nn.Identity()
    encoder.final_projection = Rearrange("b n d -> b d 1 n")
    position_embedding = PositionEmbeddingPcd(hidden_dim=hidden_dim)
    position_embedding = nn.Sequential(
        position_embedding,
    )
    model = PcdJoiner(encoder, position_embedding)
    model.num_channels = encoder.out_channels
    return model
