# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


def build_sam_vit_h(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam,
    "vit_h": build_sam,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = (128, 256, 256)  # 设置为3D元组 (depth, height, width)
    vit_patch_size = 16
    image_embedding_size = (8, 16, 16)
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=12,  # 建议值
            embed_dim=768,  # 建议值
            img_size=(128, 512, 512),  # 3D元组
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,  # 建议值
            patch_size=16,  # 建议值
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=(2, 5, 8),  # 建议值
            window_size=14,
            out_chans=256,  # 建议值
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=256,  # 建议值
            image_embedding_size=(8, 32, 32),  # 3D元组
            input_image_size=(128, 512, 512),  # 3D元组
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=20,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,  # 建议值
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,  # 建议值
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[0.0],  # 单通道调整
        pixel_std=[1.0],  # 单通道调整
    )

    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam
