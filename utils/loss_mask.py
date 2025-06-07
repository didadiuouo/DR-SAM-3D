import torch
from torch.nn import functional as F
from typing import List, Optional
import utils.misc as misc

def point_sample(input, point_coords, align_corners=False):
    """
    在给定坐标处采样张量。

    参数:
        input: [B, C, D, H, W]，输入张量
        point_coords: [B, N, 3]，归一化的3D坐标
        align_corners: bool，是否对齐角点

    返回:
        sampled: [B, C, N]，采样值
    """
    b, c, d, h, w = input.shape
    n = point_coords.shape[1]
    # 归一化坐标到 [-1, 1]
    point_coords = 2 * point_coords / torch.tensor([d - 1, h - 1, w - 1], device=point_coords.device) - 1
    # 构造grid，形状为 [B, N, 1, 1, 3]
    grid = point_coords.view(b, n, 1, 1, 3)  # [B, N, 1, 1, 3]

    grid = grid.to(input.device).float()
    input = input.float()
    sampled = F.grid_sample(input, grid, mode='bilinear', align_corners=align_corners)
    return sampled.view(b, c, n)  # [B, C, N]

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def get_uncertain_point_coords_with_randomness(
    logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):

    b, c, d, h, w = logits.shape
    num_samples = int(num_points * oversample_ratio)
    # 计算不确定性
    uncertainty_map = uncertainty_func(logits)  # [B, 1, D, H, W]
    # 归一化不确定性为概率分布
    uncertainty_map = uncertainty_map.view(b, -1)  # [B, D*H*W]
    uncertainty_map = uncertainty_map / (uncertainty_map.sum(dim=1, keepdim=True) + 1e-10)
    # 混合均匀采样和重要性采样
    uniform_indices = torch.randint(0, d * h * w, (b, int(num_samples * (1 - importance_sample_ratio))))
    importance_indices = torch.multinomial(uncertainty_map, int(num_samples * importance_sample_ratio), replacement=True)
    indices = torch.cat([uniform_indices.cuda(), importance_indices], dim=1)[:, :num_points]
    # 转换为3D坐标
    coords = torch.stack(
        [
            indices // (h * w),  # 深度
            (indices % (h * w)) // w,  # 高度
            indices % w,  # 宽度
        ],
        dim=-1,
    ).float()  # [B, num_points, 3]
    return coords

def dice_loss_jit(logits, labels, num_masks, weights=None):
    """
    计算Dice损失。

    参数:
        logits: [B, N]，预测logits
        labels: [B, N]，目标标签
        num_masks: int，掩码数量
        weights: [B, N]，可选的权重

    返回:
        loss: scalar，平均Dice损失
    """
    probs = torch.sigmoid(logits)
    intersection = (probs * labels).sum(dim=1)
    union = probs.sum(dim=1) + labels.sum(dim=1) + 1e-10
    dice = 2.0 * intersection / union
    loss = 1.0 - dice
    # if weights is not None:
    #     loss = loss * weights.sum(dim=1) / (weights.sum(dim=1) + 1e-10)
    return loss.sum() / (num_masks + 1e-10)


def sigmoid_ce_loss_jit(logits, labels, num_masks, weights=None):
    """
    计算sigmoid交叉熵损失。

    参数:
        logits: [B, N]，预测logits
        labels: [B, N]，目标标签
        num_masks: int，掩码数量
        weights: [B, N]，可选的权重

    返回:
        loss: scalar，平均损失
    """
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    if weights is not None:
        loss = loss * weights
    return loss.sum() / (num_masks + 1e-10)


def calculate_uncertainty(logits):
    """
    计算logits的不确定性（基于sigmoid概率的熵）。

    参数:
        logits: [B, num_classes, ...]，预测的logits

    返回:
        uncertainty: [B, 1, ...]，每个像素的不确定性
    """
    probs = torch.sigmoid(logits)  # [B, num_classes, D, H, W]
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1, keepdim=True)  # [B, 1, D, H, W]
    return entropy


def loss_masks(src_masks, target_masks, num_masks, oversample_ratio=3.0, num_classes=1):
    """
    Compute mask loss with weighted sampling for class imbalance in multi-label tasks.

    Args:
        src_masks: [B, num_classes, D, H, W], predicted logits for 20 classes
        target_masks: [B, num_classes, D, H, W], one-hot encoded target masks (float32)
        num_masks: int, number of masks (batch size)
        oversample_ratio: float, oversampling ratio for uncertain points
        num_classes: int, number of classes (default: 20)

    Returns:
        loss_mask: scalar, weighted cross-entropy loss
        loss_dice: scalar, weighted Dice loss
    """

    # 计算类别分布
    with torch.no_grad():
        # class_weights = [target_masks[:, c].float().mean().item() for c in range(num_classes)]
        # print(f"Class pixel ratios: {class_weights}")

        # 为每个类别采样点
        num_points = 112 * 112 * src_masks.shape[2] // 112 // 4
        point_coords_list, point_labels_list = [], []
        skipped_classes = []
        for c in range(num_classes):
            mask = target_masks[:, c:c+1]  # [B, 1, D, H, W]
            if mask.sum() > 0:
                coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    num_points // num_classes,
                    oversample_ratio,
                    0.75,
                )
                coords = coords.to(mask.device).float()  # 转换为 float32 并确保设备一致
                labels = point_sample(mask, coords, align_corners=False).squeeze(1)  # [B, N]
                point_coords_list.append(coords)
                point_labels_list.append(labels)  # 0或1，表示该类别的存在
            else:
                skipped_classes.append(f"Class {c} has no pixels, skipping; ")
        # print(" ".join(skipped_classes))
        point_coords = torch.cat(point_coords_list, dim=1)  # [B, num_points, 3]
        point_labels = torch.cat(point_labels_list, dim=1)  # [B, num_points]

    # 采样logits
    point_logits = point_sample(src_masks, point_coords, align_corners=False)  # [B, num_classes, N]

    # 加权损失
    weights = torch.tensor([0.1] + [10.0] * (num_classes - 1), device=src_masks.device)  # 假设背景权重较低
    loss_mask = 0
    loss_dice = 0
    for c in range(num_classes):
        class_logits = point_logits[:, c]  # [B, N]
        class_labels = point_labels  # [B, N]
        # class_weight = weights[c]
        loss_mask += sigmoid_ce_loss_jit(class_logits, class_labels, num_masks)
        loss_dice += dice_loss_jit(class_logits, class_labels, num_masks)
    loss_mask /= num_classes
    loss_dice /= num_classes

    return loss_mask, loss_dice


