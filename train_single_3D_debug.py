# Copyright by DrSAM team.
# All rights reserved.
# Reference from SAM and HQ-SAM, thanks to them.
import os
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
import torchio as tio
from segment_anything_3d import sam_model_registry3D
from segment_anything_3d.modeling import TwoWayTransformer, MaskDecoder3D, TwoWayTransformer3D

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
from utils.metric_utils import compute_metrics_drsam, print_computed_metrics
import utils.misc as misc
# from termcolor import colored
import copy


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


# Custom 3D LayerNorm to replace LayerNorm2d
class LayerNorm3d(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm3d, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x):
        # x: [batch, channels, depth, height, width]
        x = x.permute(0, 2, 3, 4, 1)  # [batch, depth, height, width, channels]
        x = self.ln(x)
        x = x.permute(0, 4, 1, 2, 3)  # [batch, channels, depth, height, width]
        return x


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            self.conv1 = DoubleConv(in_channels + in_channels // 4, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3=None) -> torch.Tensor:
        x1 = self.up(x1)
        # x1, x2, x3: [N, C, D, H, W]
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]

        # padding: [padding_left, padding_right, padding_front, padding_back, padding_top, padding_bottom]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2,
                        diff_z // 2, diff_z - diff_z // 2])
        if x3 is not None:
            x = torch.cat([x3, x2, x1], dim=1)
            x = self.conv1(x)
        else:
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
        return x


class DrMaskDecoder(MaskDecoder3D):
    def __init__(self, model_type, bilinear: bool = False):
        super().__init__(
            transformer_dim=384,

            num_multimask_outputs=False,
            activation=nn.GELU,
            iou_head_depth=1,
            iou_head_hidden_dim=256,
        )
        # assert model_type in ["vit_b", "vit_l", "vit_h"]

        # checkpoint_dict = {
        #     "vit_b": "pretrained_checkpoint/sam_vit_b_maskdecoder.pth",
        #     "vit_l": "pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
        #     "vit_h": "pretrained_checkpoint/sam_vit_h_maskdecoder.pth"
        # }
        # checkpoint_path = checkpoint_dict[model_type]
        # self.load_state_dict(torch.load(checkpoint_path))
        print("Dr-SAM init")
        for n, p in self.named_parameters():
            p.requires_grad = False

        transformer_dim = 384
        vit_dim_dict = {"vit_b": 768, "vit_l": 1024, "vit_h": 1280}
        vit_dim = vit_dim_dict[model_type]

        # self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = 1
        self.num_hf_tokens = 1

        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.hf_token = nn.Embedding(self.num_hf_tokens, transformer_dim)
        self.output_hypernetworks_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transformer_dim, transformer_dim),
                nn.ReLU(),
                nn.Linear(transformer_dim, transformer_dim // 8)
            ) for _ in range(self.num_mask_tokens)
        ])
        self.hf_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(transformer_dim, transformer_dim),
                nn.ReLU(),
                nn.Linear(transformer_dim, transformer_dim // 8)
            ) for _ in range(self.num_hf_tokens)
        ])


        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose3d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm3d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose3d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2)
        )

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose3d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm3d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose3d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )

        self.embedding_maskfeature = nn.Sequential(
            nn.Conv3d(transformer_dim // 8, transformer_dim // 4, kernel_size=3, stride=1, padding=1),
            LayerNorm3d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv3d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, stride=1, padding=1)
        )

        self.up_bilinear2 = nn.ConvTranspose3d(vit_dim, transformer_dim // 2, kernel_size=2, stride=2)
        self.up_bilinear4 = nn.ConvTranspose3d(vit_dim, transformer_dim // 4, kernel_size=4, stride=4)
        self.up_bilinear8 = nn.ConvTranspose3d(vit_dim, transformer_dim // 8, kernel_size=8, stride=8)

        factor = 2 if bilinear else 1
        self.up1 = Up(transformer_dim, transformer_dim // 2 // factor, bilinear)  # Assumes Up is updated for 3D
        self.up2 = Up(transformer_dim // 2, transformer_dim // 4 // factor, bilinear)
        self.up3 = Up(transformer_dim // 4, transformer_dim // 8, bilinear)
        self.down4 = nn.Sequential(
            nn.Conv3d(transformer_dim // 8, transformer_dim // 8, kernel_size=2, stride=2),
            LayerNorm3d(transformer_dim // 8),
            nn.GELU(),)

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
            med_token_only: bool = False,
            hierarchical_embeddings: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = hierarchical_embeddings[0].permute(0, 4, 1, 2, 3)
        x1 = self.up_bilinear8(x1)
        x2 = hierarchical_embeddings[1].permute(0, 4, 1, 2, 3)
        x2 = self.up_bilinear4(x2)
        x3 = hierarchical_embeddings[2].permute(0, 4, 1, 2, 3)
        x3 = self.up_bilinear2(x3)

        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                x3=x3[i_batch].unsqueeze(0),
                x2=x2[i_batch].unsqueeze(0),
                x1=x1[i_batch].unsqueeze(0),
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks, 0)
        iou_preds = torch.cat(iou_preds, 0)

        # Select the correct mask or masks for output
        if multimask_output:
            # Mask with highest IoU score
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_preds = iou_preds[:, mask_slice]
            iou_preds, max_iou_idx = torch.max(iou_preds, dim=1)
            iou_preds = iou_preds.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0)), max_iou_idx].unsqueeze(1)
        else:
            # Single mask output, default
            mask_slice = slice(0, 1)
            masks_sam = masks[:, mask_slice]
        masks_dr = masks
        # masks_dr = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens), :, :, :]

        if med_token_only:
            return masks_dr
        else:
            return masks_sam, masks_dr

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            x3: torch.Tensor = None,
            x2: torch.Tensor = None,
            x1: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:




        # Concatenate IoU, mask, and hf tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, d, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]  # [B, 384]
        mask_tokens_out = hs[:, 1:2, :]  # [B, 1, 384]，仅前景
        # Upscale mask embeddings
        src = src.transpose(1, 2).view(b, c, d, h, w)
        x4 = src
        upscaled_embedding_sam = self.output_upscaling(src)  # dim256, assume 3D upscaling

        x = self.up1(x4, x3)
        x = self.up2(x, x2, upscaled_embedding_sam)
        x = self.up3(x, x1)
        x = self.down4(x)
        upscaled_embedding_ours = x

        b, c, d, h, w = upscaled_embedding_sam.shape
        # 在 predict_masks 方法中，替换矩阵乘法和 view 的部分
        hyper_in_sam = self.output_hypernetworks_mlps[0](mask_tokens_out[:, 0, :])  # [B, 48]
        masks_sam = (hyper_in_sam @ upscaled_embedding_sam.view(b, c, d * h * w)).view(b, 1, d, h, w)

        b1, c1, d1, h1, w1 = upscaled_embedding_ours.shape
        hyper_in_ours = self.hf_mlps[0](mask_tokens_out[:, 0, :])  # [B, 48]
        masks_ours = (hyper_in_ours @ upscaled_embedding_ours.view(b1, c1, d1 * h1 * w1)).view(b1, 1, d1, h1, w1)

        # Combine masks_sam and masks_ours (element-wise average for each class)
        masks = (masks_sam + masks_ours) / 2.0

        # Predict IoU scores for 20 classes
        iou_pred = self.iou_prediction_head(iou_token_out)  # [batch, 20]

        return masks, iou_pred


def show_anns(masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    for i, (mask, iou, biou) in enumerate(zip(masks, ious, boundary_ious)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())

        plt.axis('off')
        plt.savefig(filename + '_' + str(i) + '.png', bbox_inches='tight', pad_inches=-0.1)
        plt.close()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

import nibabel as nib
def save_nii(data, filename, affine=None):
    """
    将 PyTorch 张量保存为 NII 格式文件
    Args:
        data: PyTorch 张量，形状 [batch, 1, D, H, W] 或 [D, H, W]
        filename: 输出文件路径（例如 'label.nii.gz'）
        affine: 仿射矩阵（可选，默认使用单位矩阵）
    """
    # 确保数据是 numpy 数组
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # 移除 batch 和 channel 维度（如果存在）
    if data.ndim == 5:
        data = data[0, 0]  # 提取 [D, H, W]
    elif data.ndim == 4:
        data = data[0]  # 提取 [D, H, W]

    # 确保数据类型为 float32 或 int16（NII 格式常用）
    data = data.astype(np.float32)

    # 创建仿射矩阵（如果未提供，默认单位矩阵）
    if affine is None:
        affine = np.eye(4)

    # 创建 NIfTI 图像
    nii_img = nib.Nifti1Image(data, affine)

    # 保存到文件
    nib.save(nii_img, filename)
    print(f"已保存 NII 文件: {filename}")


# 示例：在训练循环中保存真实标签和预测结果
def save_labels_and_predictions(masks_dr_binary, labels, epoch, output_dir="nii_output"):
    """
    保存预测结果和真实标签为 NII 格式
    Args:
        masks_dr_binary: 预测结果，形状 [batch, 1, D, H, W]
        labels: 真实标签，形状 [batch, 1, D, H, W]
        epoch: 当前轮数
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存真实标签
    label_filename = os.path.join(output_dir, f"label_sample_{epoch}.nii.gz")
    save_nii(labels, label_filename)

    # 保存预测结果
    pred_filename = os.path.join(output_dir, f"pred_sample_{epoch}.nii.gz")
    save_nii(masks_dr_binary, pred_filename)







from scipy.ndimage import binary_dilation, binary_erosion


def compute_iou(preds: torch.Tensor, target: torch.Tensor) -> dict:
    num_classes = 20
    class_iou = [0.0] * num_classes  # 初始化所有类别的 IoU 为 0
    class_count = [0] * num_classes
    iou_sum = 0.0
    num_valid = 0
    skipped_classes = []
    # Ensure target matches pred's spatial dimensions

    # # Upsample to match target's spatial dimensions
    # if preds.shape[2:] != target.shape[2:]:
    #     conv_channel = nn.Conv2d(32, 12, kernel_size=1).to(preds.device)
    #     preds = conv_channel(preds.float().to(preds.device))
    #     preds = F.interpolate(preds, size=target.shape[2:], mode='bilinear',
    #                           align_corners=False)  # Shape: (1, 12, 880, 880)

    pred = preds.long()
    target = target.long()

    # 获取当前样本的有效类别
    valid_classes = torch.unique(target).cpu().numpy()

    for c in valid_classes:

        intersection = ((pred == c) & (target == c)).float()
        intersection = intersection.sum((1, 2, 3))  # [B]
        union = ((pred == c) | (target == c)).float()
        union = union.sum((1, 2, 3))  # [B]
        # print(f"Class {c}: intersection={intersection.sum().item()}, union={union.sum().item()}")
        if union.sum() == 0:
            skipped_classes.append(f"Class {c} has no pixels, skipping")
            class_iou[c] = 0.0
            class_count[c] = 0
            continue
        iou_c = intersection / (union + 1e-6)  # [B]
        class_iou[c] = iou_c.mean().item()
        class_count[c] = (union > 0).sum().item()
        iou_sum += iou_c.sum().item()
        num_valid += (union > 0).sum().item()

    if skipped_classes:
        print(" ".join(skipped_classes))

    mean_iou = iou_sum / num_valid if num_valid > 0 else 0.0

    return {
        'class_iou': class_iou,
        'mean_iou': mean_iou,
        # 'valid_classes': valid_classes
    }


def extract_boundary(mask, kernel_size=3):
    """
    提取掩码边界。
    输入: mask [D, H, W]，二值掩码（单类）
    输出: boundary [D, H, W]，二值边界掩码
    """
    mask_np = mask.cpu().numpy().astype(bool)
    dilated = binary_dilation(mask_np, structure=np.ones((kernel_size, kernel_size, kernel_size)))
    eroded = binary_erosion(mask_np, structure=np.ones((kernel_size, kernel_size, kernel_size)))
    boundary = dilated ^ eroded
    return torch.tensor(boundary, dtype=torch.float32, device=mask.device)


def compute_boundary_iou(preds, target, num_classes=1, kernel_size=3):
    """
    计算 Boundary IoU。
    输入: preds [B, 1, D, H, W], target [B, 1, D, H, W]，整数类型
    输出: boundary_iou [num_classes]，每个类别的平均 Boundary IoU
    """
    preds = preds.squeeze(1)  # [B, D, H, W]
    target = target.squeeze(1)  # [B, D, H, W]
    boundary_iou = torch.zeros(num_classes, device=preds.device)
    for cls in range(num_classes):
        batch_iou = 0.0
        valid_batches = 0
        for i in range(len(preds)):
            pred_cls = (preds[i] == cls).float()
            target_cls = (target[i] == cls).float()
            if target_cls.sum() == 0 and pred_cls.sum() == 0:
                continue  # 跳过空类
            pred_boundary = extract_boundary(pred_cls, kernel_size)
            target_boundary = extract_boundary(target_cls, kernel_size)
            intersection = (pred_boundary * target_boundary).sum()
            union = (pred_boundary + target_boundary - pred_boundary * target_boundary).sum()
            if union > 0:
                batch_iou += intersection / (union + 1e-8)
                valid_batches += 1
        boundary_iou[cls] = batch_iou / (valid_batches + 1e-8)
    return boundary_iou


def compute_dice(preds, target, num_classes=20):
    """
    计算 Dice 分数。
    输入: preds [B, 1, D, H, W], target [B, 1, D, H, W]，整数类型
    输出: dice [num_classes]，每个类别的平均 Dice 分数
    """
    # # Upsample to match target's spatial dimensions
    # if preds.shape[2:] != target.shape[2:]:
    #     conv_channel = nn.Conv2d(32, 12, kernel_size=1).to(preds.device)
    #     preds = conv_channel(preds.float().to(preds.device))
    #     preds = F.interpolate(preds, size=target.shape[2:], mode='bilinear',
    #                           align_corners=False)  # Shape: (1, 12, 880, 880)

    preds = preds.squeeze(1)  # [B, D, H, W]
    target = target.squeeze(1)  # [B, D, H, W]

    dice = torch.zeros(num_classes, device=preds.device)
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum(dim=(1, 2, 3))
        dice[cls] = (2 * intersection / (pred_cls.sum(dim=(1, 2, 3)) + target_cls.sum(dim=(1, 2, 3)) + 1e-8)).mean()
    return dice

def init_checkpoint(model, optimizer, lr_scheduler, ckp_path, device):
    last_ckpt = None
    if os.path.exists(ckp_path):
        last_ckpt = torch.load(ckp_path, map_location=device, weights_only=False)

    if last_ckpt:
        model.load_state_dict(last_ckpt['model_state_dict'])
        # start_epoch = last_ckpt['epoch']
        # optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
        # lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
        # losses = last_ckpt['losses']
        # dices = last_ckpt['dices']
        # best_loss = last_ckpt['best_loss']
        # best_dice = last_ckpt['best_dice']
        print(f"Loaded checkpoint from {ckp_path} ")
    else:
        start_epoch = 0
        print(f"No checkpoint found at {ckp_path}, start training from scratch")

def get_dicewithiou_score(prev_masks, gt3D):

    def compute_dice(mask_pred, mask_gt):
        mask_threshold = 0.5

        mask_pred = (mask_pred > mask_threshold)
        mask_gt = (mask_gt > 0)

        volume_sum = mask_gt.sum() + mask_pred.sum()
        if volume_sum == 0:
            return np.NaN
        volume_intersect = (mask_gt & mask_pred).sum()
        return 2 * volume_intersect / volume_sum

    def compute_iou(mask_pred, mask_gt):
        mask_pred = (mask_pred > 0.5)
        mask_gt = (mask_gt > 0)

        intersection = (mask_gt & mask_pred).sum()
        union = (mask_gt | mask_pred).sum()
        if union == 0:
            return np.NaN
        return intersection / union

    pred_masks = (prev_masks > 0.5)
    true_masks = (gt3D > 0)
    dice_list = []
    for i in range(true_masks.shape[0]):
        dice_list.append(compute_dice(pred_masks[i], true_masks[i]))
    dice_score = sum(dice_list) / len(dice_list)

    # Compute IoU score
    iou_list = []
    for i in range(true_masks.shape[0]):
        iou_list.append(compute_iou(pred_masks[i], true_masks[i]))

    iou_score = sum(iou_list) / len(iou_list)

    return dice_score.item(), iou_score.item()

# def save_checkpoint(self, epoch, state_dict, describe="last"):
#     torch.save(
#         {
#             "epoch": epoch + 1,
#             "model_state_dict": state_dict,
#             "optimizer_state_dict": self.optimizer.state_dict(),
#             "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
#             "losses": self.losses,
#             "dices": self.dices,
#             "best_loss": self.best_loss,
#             "best_dice": self.best_dice,
#             "args": self.args,
#             "used_datas": img_datas,
#         }, join(MODEL_SAVE_PATH, f"sam_model_{describe}.pth"))

def main(net, sam, train_datasets, valid_datasets, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                               my_transforms=[
                                                                   RandomHFlip(),
                                                                   LargeScaleJitter()
                                                               ],
                                                               batch_size=args.batch_size_train,
                                                               training=True)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                           my_transforms=[
                                                               Resize(args.input_size)
                                                           ],
                                                           batch_size=args.batch_size_valid,
                                                           training=False)
    print(len(valid_dataloaders), " valid dataloaders created")

    if torch.cuda.is_available():
        net.cuda()

    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.AdamW(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-02,
                                weight_decay=0.1)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop_epoch)
        lr_scheduler.last_epoch = args.start_epoch

        # optimizer_sam = optim.AdamW(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-02,
        #                             weight_decay=0.0001)
        optimizer_sam = torch.optim.AdamW(
            [
                {
                    'params': sam.image_encoder.parameters()
                },  # , 'lr': self.args.lr * 0.1},
                {
                    'params': sam.prompt_encoder.parameters(),
                    'lr': args.learning_rate * 0.1
                },
                {
                    'params': sam.mask_decoder.parameters(),
                    'lr': args.learning_rate * 0.1
                },
            ],
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0001)
        lr_scheduler_sam = torch.optim.lr_scheduler.StepLR(optimizer_sam, args.lr_drop_epoch)
        lr_scheduler_sam.last_epoch = args.start_epoch

        init_checkpoint(model=sam, optimizer=optimizer_sam, lr_scheduler=lr_scheduler_sam, ckp_path=args.checkpoint, device=args.device)

        train(args, net, sam, optimizer, optimizer_sam, train_dataloaders, valid_dataloaders, lr_scheduler,
              lr_scheduler_sam)
    else:
        sam = sam_model_registry3D[args.model_type](checkpoint=None)
        _ = sam.to(device=args.device)
        optimizer_sam = torch.optim.AdamW(
            [
                {
                    'params': sam.image_encoder.parameters()
                },  # , 'lr': self.args.lr * 0.1},
                {
                    'params': sam.prompt_encoder.parameters(),
                    'lr': args.learning_rate * 0.1
                },
                {
                    'params': sam.mask_decoder.parameters(),
                    'lr': args.learning_rate * 0.1
                },
            ],
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.0001)
        lr_scheduler_sam = torch.optim.lr_scheduler.StepLR(optimizer_sam, args.lr_drop_epoch)
        lr_scheduler_sam.last_epoch = args.start_epoch

        init_checkpoint(model=sam, optimizer=optimizer_sam, lr_scheduler=lr_scheduler_sam, ckp_path=args.checkpoint, device=args.device)
        if args.restore_model:
            print("restore model from:", args.restore_model)
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(args.restore_model))
            else:
                net.load_state_dict(torch.load(args.restore_model, map_location="cpu"))

        evaluate(args, net, sam, valid_dataloaders, args.visualize)


def train(args, net, sam, optimizer, optimizer_sam, train_dataloaders, valid_dataloaders, lr_scheduler,
          lr_scheduler_sam):
    os.makedirs(args.output, exist_ok=True)

    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num

    net.train()
    net = net.to(device=args.device)

    sam.train()
    sam = sam.to(device=args.device)

    for epoch in range(epoch_start, epoch_num):
        print("epoch:   ", epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
        metric_logger = misc.MetricLogger(delimiter="  ")
        num_sample = 0
        epoch_dice = 0
        epoch_iou = 0
        with tqdm(metric_logger.log_every(train_dataloaders, 100), total=len(train_dataloaders),
                  desc=f"Epoch {epoch}/{epoch_num - 1}") as pbar:
            for data in pbar:
                inputs, labels = data['image'], data['label']
                labels = labels.long()  # 转换为整数
                inputs = inputs.to(device=args.device)
                labels = labels.to(device=args.device)
                # print("ori_unique shape:", labels.shape, "ori_unique values:", torch.unique(labels))
                # 输入提示（3D）
                input_keys = ['box', 'point', 'noise_mask']
                labels_box = misc.masks_to_boxes(labels[:,0,:,:])
                try:
                    labels_points = misc.masks_sample_points(labels[:, 0, :, :, :])  # 3D 点 [B, N, 3]
                except:
                    input_keys = ['box', 'noise_mask']
                labels_512 = F.interpolate(labels.float(), size=(128, 128, 128), mode='nearest').long()
                labels_noisemask = misc.masks_noise(labels_512)  # [B, 1, D, H, W]
                # print("labels_ori shape:", labels.shape, "unique values:", torch.unique(labels))
                batched_input = []

                for b_i in range(inputs.shape[0]):
                    dict_input = dict()
                    img = inputs[b_i]  # [C, D, H, W]，C=1
                    dict_input['image'] = img.contiguous()  # [1, D, H, W]

                    input_type = random.choice(input_keys)
                    if input_type == 'box':
                        dict_input['boxes'] = labels_box[b_i:b_i + 1]  # [1, N, 6]
                    elif input_type == 'point':
                        point_coords = labels_points[b_i:b_i + 1]  # [1, N, 3]
                        dict_input['point_coords'] = point_coords
                        dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,
                                                     :]  # [1, N]
                    elif input_type == 'noise_mask':
                        dict_input['mask_inputs'] = labels_noisemask[b_i:b_i + 1]  # [1, 1, D, H, W]
                    else:
                        raise NotImplementedError
                    dict_input['original_size'] = inputs[b_i].shape[1:]  # [C, D, H, W]

                    # dict_input['original_size'] = inputs[b_i].shape[:2]
                    batched_input.append(dict_input)

                batched_output, hierarchical_embeddings = sam(batched_input, multimask_output=False)

                batch_len = len(batched_output)
                encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
                image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
                sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
                dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

                masks_sam, masks_dr = net(
                    image_embeddings=encoder_embedding,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    med_token_only=False,
                    hierarchical_embeddings=hierarchical_embeddings,
                )

                if masks_dr.shape[2:] != labels.shape[2:]:
                    masks_dr = F.interpolate(masks_dr, size=labels.shape[2:], mode='trilinear', align_corners=False)  # 形状: [1, 1, 12, 880, 880]
                # print("masks_dr unique:", torch.unique(masks_dr))
                loss_mask, loss_dice = loss_masks(masks_dr, labels, len(masks_dr))
                loss = loss_mask + loss_dice

                dice, iou = get_dicewithiou_score(prev_masks=masks_dr, gt3D=labels)

                epoch_dice +=dice
                epoch_iou +=iou

                optimizer.zero_grad()
                optimizer_sam.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_sam.step()
                num_sample +=1
                # masks_softmax = torch.softmax(masks_sam, dim=1)
                # preds = masks_softmax.argmax(dim=1, keepdim=True)
                # print("Pred classes:", preds.unique().cpu().numpy())
                if num_sample % 1 == 0:

                    # print("masks_dr unique:", torch.unique(masks_dr))

                    if labels.shape[1] > 1:
                        labels = torch.argmax(labels, dim=1, keepdim=True).float()  # 如果 labels 是 one-hot
                    else:
                        labels = labels.float()  # 保持单通道
                    # print("labels_ori shape:", labels.shape)
                    # print("labels_ori unique:", torch.unique(labels))

                    print_dice, print_iou = get_dicewithiou_score(prev_masks=masks_dr, gt3D=labels)
                    print(f'print_dice: {print_dice},   print_iou: {print_iou}')

            metric_logger.update(training_loss=loss.item(), loss_mask=loss_mask.item(), loss_dice=loss_dice.item())

        # print("Finished epoch:      ", epoch)
        # metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        lr_scheduler.step()
        # 在训练循环中，优化器更新后
        optimizer.step()
        print(f"Net parameter norm: {sum(p.norm().item() for p in net.parameters())}")

        test_stats = evaluate(args, net, sam, valid_dataloaders)
        train_stats.update(test_stats)

        net.train()

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_" + str(epoch) + ".pth"
            # print('come here save at', args.output + model_name)
            misc.save_on_master(net.state_dict(), args.output + '/net' + model_name)
            misc.save_on_master(sam.state_dict(), args.output + '/sam' + model_name)
            # torch.save(sam.state_dict(), f'sam_epoch_{epoch}.pth')
            print('come here save at', args.output)

    # Finish training
    print("Training Reaches The Maximum Epoch Number")

    # # merge sam and DrSAM
    # sam_ckpt = torch.load(args.checkpoint)
    # hq_decoder = torch.load(args.output + model_name)
    # for key in hq_decoder.keys():
    #     sam_key = 'mask_decoder.'+key
    #     if sam_key not in sam_ckpt.keys():
    #         sam_ckpt[sam_key] = hq_decoder[key]
    # model_name = "/drsam_epoch_"+str(epoch)+".pth"
    # torch.save(sam_ckpt, args.output + model_name)


def evaluate(args, net, sam, valid_dataloaders, visualize=False):
    net.eval()
    sam.eval()
    print("Validating...")
    test_stats = {}
    i=0
    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print(f"Valid_dataloader len: {len(valid_dataloader)}")

        # for data_val in metric_logger.log_every(valid_dataloader, 100):
        for data_val in tqdm(metric_logger.log_every(valid_dataloader, 100), total=len(valid_dataloader)):
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori = data_val['imidx'], data_val['image'], data_val[
                'label'], data_val['shape'], data_val['ori_label']
            inputs_val = inputs_val.to(device=args.device)
            labels_val = labels_val.to(device=args.device).long()
            labels_ori = labels_ori.to(device=args.device).long()

            # inputs, labels = data['image'], data['label']
            labels_ori = labels_ori.long()  # 转换为整数
            inputs = inputs_val.to(device=args.device)
            labels = labels_ori.to(device=args.device)

            # 这是一个尝试将MED3d的评价指标计算过程迁移到DRmed上时，发现的问题，该评价指标为单标签验证方式，结合标签信息验证模型。
            # 模型输入为image和mask的nii.path信息。

            # unique_label = np.unique(labels)
            # exist_categories = [int(l) for l in unique_label if l != 0]
            # meta_info = labels.shape[2:]
            #
            # # 将 inputs_val 和 labels_ori 转换为 TorchIO 的 ScalarImage 和 LabelMap
            # image = tio.ScalarImage(tensor=inputs_val)  # 假设 inputs_val 是图像数据
            # label = tio.LabelMap(tensor=labels_ori)  # 假设 labels_ori 是标签数据
            #
            # # 创建一个 Subject 对象
            # subject = tio.Subject(image=image, label=label)
            #
            # for category_index in exist_categories:
            #     category_specific_subject = copy.deepcopy(subject)
            #     category_specific_meta_info = copy.deepcopy(meta_info)
            #
            #     roi_image, roi_label, meta_info = data_preprocess(category_specific_subject,
            #                                                       category_specific_meta_info,
            #                                                       category_index=category_index,
            #                                                       target_spacing=target_spacing,
            #                                                       crop_size=crop_size)


            # 输入提示（3D）
            input_keys = ['box', 'point', 'noise_mask']
            labels_box = misc.masks_to_boxes(labels[:, 0, :, :])
            try:
                labels_points = misc.masks_sample_points(labels[:, 0, :, :, :])  # 3D 点 [B, N, 3]
            except:
                input_keys = ['box', 'noise_mask']
            labels_512 = F.interpolate(labels.float(), size=(128, 128, 128), mode='nearest').long()
            labels_noisemask = misc.masks_noise(labels_512)  # [B, 1, D, H, W]

            batched_input = []
            for b_i in range(inputs.shape[0]):
                dict_input = dict()
                img = inputs[b_i]  # [C, D, H, W]，C=1
                dict_input['image'] = img.contiguous()  # [1, D, H, W]

                input_type = random.choice(input_keys)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[b_i:b_i + 1]  # [1, N, 6]
                elif input_type == 'point':
                    point_coords = labels_points[b_i:b_i + 1]  # [1, N, 3]
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,
                                                 :]  # [1, N]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[b_i:b_i + 1]  # [1, 1, D, H, W]
                else:
                    raise NotImplementedError
                dict_input['original_size'] = inputs[b_i].shape[1:]  # [C, D, H, W]

                # dict_input['original_size'] = inputs[b_i].shape[:2]
                batched_input.append(dict_input)

            # 检查输入是否有效
            for input_dict in batched_input:
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                        print(f"警告: {key} 包含 nan")

            batched_output, hierarchical_embeddings = sam(batched_input, multimask_output=False)

            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

            masks_sam, masks_dr = net(image_embeddings=encoder_embedding, image_pe=image_pe, sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings, multimask_output=False, med_token_only=False, hierarchical_embeddings=hierarchical_embeddings,)

            masks_dr = torch.sigmoid(masks_dr)  # 形状: [1, 1, 32, 32, 32]
            print("masks_dr shape:", masks_dr.shape, "unique sigmoid values:", torch.unique(masks_dr))

            # 上采样到目标形状
            if masks_dr.shape[2:] != labels.shape[2:]:
                masks_dr = F.interpolate(masks_dr, size=labels.shape[2:], mode='trilinear',
                                         align_corners=False)  # 形状: [1, 1, 12, 880, 880]

            # print("masks_dr unique:", torch.unique(masks_dr))

            print_dice, print_iou = get_dicewithiou_score(prev_masks=masks_dr, gt3D=labels)
            print(f'print_dice: {print_dice},  print_iou: {print_iou}')
            if args.visualize:

                i += 1
                # 保存 NII 文件（仅保存第一个 batch 以节省空间）
                save_labels_and_predictions(masks_dr, labels, i)

        print('============================')
        metric_logger.synchronize_between_processes()
        print(f"Averaged stats: {metric_logger}")
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)

    return test_stats


def get_args_parser():
    parser = argparse.ArgumentParser('DrSAM', add_help=False)

    parser.add_argument("--output", type=str, default='work_dirs/DrSAM_b',
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--model-type", type=str, default="vit_b",
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--checkpoint", type=str, default='checkpoint/sam_med3d_turbo.pth',#'work_dirs/DrSAM_b/sam/epoch_130.pth'
                        help="The path to the SAM checkpoint to use for mask generation.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--learning_rate', default=1e-6, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--lr_drop_epoch', default=20, type=int)
    parser.add_argument('--max_epoch_num', default=500, type=int)
    parser.add_argument('--input_size', default=[12, 880, 880], type=list)
    parser.add_argument('--batch_size_train', default=1, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=10, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')
    # if you want to eval or visulize
    parser.add_argument('--eval', default=False)
    parser.add_argument('--visualize', default=False)
    parser.add_argument("--restore-model", type=str, default= 'work_dirs/DrSAM_b/net/epoch_50.pth',#'work_dirs/DrSAM_b/net/epoch_0.pth'
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()

if __name__ == "__main__":
    ### --------------- Configuring the Train and Valid datasets ---------------

    vessel_ours = {"name": "vessel_ours",
                   "im_dir": "./data/vessel/train/img",
                   "gt_dir": "./data/vessel/train/msk",
                   "im_ext": ".nii.gz",
                   "gt_ext": ".nii.gz"}

    lumbar_spine_MRi = {"name": "lumbar_spine_MRi",
                   "im_dir": r"D:\Desktop\3D_segmentation\3D-DRSAM\3DDrsam_data\train\imagesTr",
                   "gt_dir": r"D:\Desktop\3D_segmentation\3D-DRSAM\3DDrsam_data\train\labelsTr",
                   "im_ext": ".nii.gz",
                   "gt_ext": ".nii.gz"}



    # valid set
    lumbar_spine_MRi_val = {"name": "lumbar_spine_MRi_val",
                       "im_dir": r"D:\Desktop\3D_segmentation\3D-DRSAM\3DDrsam_data\val\img",
                       "gt_dir": r"D:\Desktop\3D_segmentation\3D-DRSAM\3DDrsam_data\val\msk",
                       "im_ext": ".nii.gz",
                       "gt_ext": ".nii.gz"}
    # valid set

    vessel_ours_val = {"name": "vessel_ours_val",
                       "im_dir": "./data/vessel/val/img",
                       "gt_dir": "./data/vessel/val/msk",
                       "im_ext": ".nii.gz",
                       "gt_ext": ".nii.gz"}


    train_datasets = [lumbar_spine_MRi, ]
    valid_datasets = [lumbar_spine_MRi_val, ]

    args = get_args_parser()
    net = DrMaskDecoder(args.model_type)

    sam = sam_model_registry3D[args.model_type](checkpoint=None)

    main(net, sam, train_datasets, valid_datasets, args)
