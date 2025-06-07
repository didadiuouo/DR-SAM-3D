# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm3d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
# 3D版本的ImageEncoderViT
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (128, 256, 256),  # (depth, height, width)
        patch_size: int = 16,
        in_chans: int = 1,  # 单通道灰度医学图像
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Args:
            img_size (Tuple[int, int, int]): 输入体视图像尺寸 (depth, height, width)。
            patch_size (int): Patch尺寸（假设在D, H, W上相同）。
            in_chans (int): 输入图像通道数（默认1，灰度医学图像）。
            embed_dim (int): Patch嵌入维度。
            depth (int): ViT深度。
            num_heads (int): 每个ViT块的注意力头数。
            mlp_ratio (float): MLP隐藏维度与嵌入维度的比例。
            qkv_bias (bool): 是否为query、key、value添加可学习偏置。
            norm_layer (nn.Module): 归一化层。
            act_layer (nn.Module): 激活层。
            use_abs_pos (bool): 是否使用绝对位置嵌入。
            use_rel_pos (bool): 是否在注意力图中添加相对位置嵌入。
            rel_pos_zero_init (bool): 是否对相对位置参数进行零初始化。
            window_size (int): 窗口注意力块的窗口尺寸（3D）。
            global_attn_indexes (list): 使用全局注意力的块索引。
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed3D(
            kernel_size=(patch_size, patch_size, patch_size),
            stride=(patch_size, patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # 初始化3D绝对位置嵌入
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size[0] // patch_size, img_size[1] // patch_size, img_size[2] // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block3D(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size[0] // patch_size, img_size[1] // patch_size, img_size[2] // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv3d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm3d(out_chans),
            nn.Conv3d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm3d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Args:
            x: 输入张量，形状为 [B, C, D, H, W]
        Returns:
            x: 输出特征图，形状为 [B, out_chans, D//patch_size, H//patch_size, W//patch_size]
            hierarchical_embeddings: 层次嵌入列表，每个元素形状为 [B, D//patch_size, H//patch_size, W//patch_size, embed_dim]
        """
        x = self.patch_embed(x)  # [B, D//patch_size, H//patch_size, W//patch_size, embed_dim]
        if self.pos_embed is not None:
            x = x + self.pos_embed

        hierarchical_embeddings = []
        for blk in self.blocks:
            x = blk(x)
            if blk.window_size == 0:
                hierarchical_embeddings.append(x)

        x = self.neck(x.permute(0, 4, 1, 2, 3))  # [B, D//patch_size, H//patch_size, W//patch_size, embed_dim] -> [B, out_chans, D//patch_size, H//patch_size, W//patch_size]
        return x, hierarchical_embeddings

class Block3D(nn.Module):
    """3D Transformer块，支持窗口注意力和残差传播"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): 输入通道数。
            num_heads (int): 注意力头数。
            mlp_ratio (float): MLP隐藏维度与嵌入维度的比例。
            qkv_bias (bool): 是否为query、key、value添加偏置。
            norm_layer (nn.Module): 归一化层。
            act_layer (nn.Module): 激活层。
            use_rel_pos (bool): 是否使用相对位置嵌入。
            rel_pos_zero_init (bool): 是否对相对位置参数零初始化。
            window_size (int): 3D窗口注意力尺寸。
            input_size (Tuple): 输入分辨率 (D, H, W)。
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention3D(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        # 3D窗口分区
        if self.window_size > 0:
            D, H, W = x.shape[1], x.shape[2], x.shape[3]
            x, pad_dhw = window_partition_3d(x, self.window_size)

        x = self.attn(x)
        # 反转窗口分区
        if self.window_size > 0:
            x = window_unpartition_3d(x, self.window_size, pad_dhw, (D, H, W))

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class Attention3D(nn.Module):
    """3D多头注意力块，支持相对位置嵌入"""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): 输入通道数。
            num_heads (int): 注意力头数。
            qkv_bias (bool): 是否为query、key、value添加偏置。
            use_rel_pos (bool): 是否使用相对位置嵌入。
            rel_pos_zero_init (bool): 是否对相对位置参数零初始化。
            input_size (Tuple): 输入分辨率 (D, H, W)。
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None, "必须提供input_size以使用相对位置编码"
            # 初始化3D相对位置嵌入
            self.rel_pos_d = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[2] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, H, W, _ = x.shape
        # qkv形状: (3, B, nHead, D * H * W, C)
        qkv = self.qkv(x).reshape(B, D * H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v形状: (B * nHead, D * H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, D * H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos_3d(attn, q, self.rel_pos_d, self.rel_pos_h, self.rel_pos_w, (D, H, W), (D, H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, D, H, W, -1).permute(0, 2, 3, 4, 1, 5).reshape(B, D, H, W, -1)
        x = self.proj(x)
        return x

def window_partition_3d(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    将3D输入分区为非重叠的3D窗口，必要时填充。
    Args:
        x: 输入张量，形状为 [B, D, H, W, C]。
        window_size: 3D窗口尺寸。
    Returns:
        windows: 分区后的窗口，形状为 [B * num_windows, window_size, window_size, window_size, C]。
        (Dp, Hp, Wp): 填充前的深度、高度和宽度。
    """
    B, D, H, W, C = x.shape

    pad_d = (window_size - D % window_size) % window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

    x = x.view(B, Dp // window_size, window_size, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows, (Dp, Hp, Wp)

def window_unpartition_3d(
    windows: torch.Tensor, window_size: int, pad_dhw: Tuple[int, int, int], dhw: Tuple[int, int, int]
) -> torch.Tensor:
    """
    将3D窗口反分区为原始序列并移除填充。
    Args:
        windows: 输入张量，形状为 [B * num_windows, window_size, window_size, window_size, C]。
        window_size: 窗口尺寸。
        pad_dhw: 填充前的深度、高度和宽度 (Dp, Hp, Wp)。
        dhw: 原始深度、高度和宽度 (D, H, W)。
    Returns:
        x: 反分区后的序列，形状为 [B, D, H, W, C]。
    """
    Dp, Hp, Wp = pad_dhw
    D, H, W = dhw
    B = windows.shape[0] // (Dp * Hp * Wp // window_size // window_size // window_size)
    x = windows.view(B, Dp // window_size, Hp // window_size, Wp // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, Dp, Hp, Wp, -1)

    if Dp > D or Hp > H or Wp > W:
        x = x[:, :D, :H, :W, :].contiguous()
    return x

def get_rel_pos_3d(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    根据query和key的相对位置获取3D相对位置嵌入。
    Args:
        q_size: query尺寸。
        k_size: key尺寸。
        rel_pos: 相对位置嵌入，形状为 (L, C)。
    Returns:
        按相对位置提取的位置嵌入。
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos_3d(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_d: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int, int],
    k_size: Tuple[int, int, int],
) -> torch.Tensor:
    """
    计算3D分解相对位置嵌入。
    Args:
        attn: 注意力图。
        q: query张量，形状为 (B, q_d * q_h * q_w, C)。
        rel_pos_d: 深度轴的相对位置嵌入 (Ld, C)。
        rel_pos_h: 高度轴的相对位置嵌入 (Lh, C)。
        rel_pos_w: 宽度轴的相对位置嵌入 (Lw, C)。
        q_size: query的空间序列尺寸 (q_d, q_h, q_w)。
        k_size: key的空间序列尺寸 (k_d, k_h, k_w)。
    Returns:
        attn: 添加了相对位置嵌入的注意力图。
    """
    q_d, q_h, q_w = q_size
    k_d, k_h, k_w = k_size
    Rd = get_rel_pos_3d(q_d, k_d, rel_pos_d)
    Rh = get_rel_pos_3d(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos_3d(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_d, q_h, q_w, dim)
    rel_d = torch.einsum("bdhwc,dkc->bdhwk", r_q, Rd)
    rel_h = torch.einsum("bdhwc,hkc->bdhwk", r_q, Rh)
    rel_w = torch.einsum("bdhwc,wkc->bdhwk", r_q, Rw)

    attn = (
        attn.view(B, q_d, q_h, q_w, k_d, k_h, k_w)
        + rel_d[:, :, :, :, :, None, None]
        + rel_h[:, :, :, :, None, :, None]
        + rel_w[:, :, :, :, None, None, :]
    ).view(B, q_d * q_h * q_w, k_d * k_h * k_w)

    return attn

class PatchEmbed3D(nn.Module):
    """3D图像到Patch嵌入"""
    def __init__(
        self,
        kernel_size: Tuple[int, int, int] = (16, 16, 16),
        stride: Tuple[int, int, int] = (16, 16, 16),
        padding: Tuple[int, int, int] = (0, 0, 0),
        in_chans: int = 1,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size: 投影层的核尺寸。
            stride: 投影层的步幅。
            padding: 投影层的填充。
            in_chans: 输入图像通道数。
            embed_dim: Patch嵌入维度。
        """
        super().__init__()
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, C, D, H, W] -> [B, embed_dim, D//patch_size, H//patch_size, W//patch_size]
        x = x.permute(0, 2, 3, 4, 1)  # [B, embed_dim, D//patch_size, H//patch_size, W//patch_size] -> [B, D//patch_size, H//patch_size, W//patch_size, embed_dim]
        return x

