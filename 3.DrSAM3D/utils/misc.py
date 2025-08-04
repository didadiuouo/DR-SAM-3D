# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import random
import subprocess
import time
from collections import OrderedDict, defaultdict, deque
import datetime
import pickle
from typing import Optional, List

import json, time
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

import colorsys
import torch.nn.functional as F

import cv2

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        if d.shape[0] == 0:
            return 0
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            # print(name, str(meter))
            # import ipdb;ipdb.set_trace()
            if meter.count > 0:
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, logger=None):
        if logger is None:
            print_func = print
        else:
            print_func = logger.info

        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 512.0 * 512.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj

            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print_func(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print_func(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print_func('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message




def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'WORLD_SIZE' in os.environ and os.environ['WORLD_SIZE'] != '': # 'RANK' in os.environ and
        # args.rank = int(os.environ["RANK"])
        # args.world_size = int(os.environ['WORLD_SIZE'])
        # args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])

        # launch by torch.distributed.launch
        # Single node
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 1 --rank 0 ...
        # Multi nodes
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 0 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        #   python -m torch.distributed.launch --nproc_per_node=8 main.py --world-size 2 --rank 1 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' ...
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.gpu = args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print(json.dumps(dict(os.environ), indent=2))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.local_rank = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NPROCS'])

        print('world size: {}, world rank: {}, local rank: {}, device_count: {}'.format(args.world_size, args.rank, args.local_rank, torch.cuda.device_count()))
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    print("world_size:{} rank:{} local_rank:{}".format(args.world_size, args.rank, args.local_rank))
    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'gloo'
    print('| distributed init (rank {}): {}'.format(args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    print("Before torch.distributed.barrier()")
    torch.distributed.barrier()
    print("End torch.distributed.barrier()")
    setup_for_distributed(args.rank == 0)


def masks_to_boxes(masks): #med3d未修改，这里我修改成3D提示框
    """
        从 3D 掩码生成 3D 边界框。
        输入: masks [N, D, H, W]，整数类型，表示分割掩码
        输出: boxes [N, 6]，浮点类型，[x_min, y_min, z_min, x_max, y_max, z_max]
        """
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)

    # 确保输入是 4D: [N, D, H, W]
    if masks.dim() != 4:
        raise ValueError(f"期望 4D 输入 [N, D, H, W]，实际形状 {masks.shape}")

    n, d, h, w = masks.shape

    # 创建深度、高度、宽度的坐标网格
    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    z, y, x = torch.meshgrid(z, y, x, indexing='ij')

    # 二值化掩码（医学影像掩码阈值 > 0）
    mask_binary = masks > 0

    # 计算最小/最大坐标
    x_mask = mask_binary * x.unsqueeze(0)  # [N, D, H, W]
    x_max = x_mask.flatten(1).max(-1)[0]  # [N]
    x_min = x_mask.masked_fill(~mask_binary, 1e8).flatten(1).min(-1)[0]  # [N]

    y_mask = mask_binary * y.unsqueeze(0)
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~mask_binary, 1e8).flatten(1).min(-1)[0]

    z_mask = mask_binary * z.unsqueeze(0)
    z_max = z_mask.flatten(1).max(-1)[0]
    z_min = z_mask.masked_fill(~mask_binary, 1e8).flatten(1).min(-1)[0]

    # 处理空掩码（无前景像素）
    x_max = torch.where(x_max == 0, torch.ones_like(x_max), x_max)
    y_max = torch.where(y_max == 0, torch.ones_like(y_max), y_max)
    z_max = torch.where(z_max == 0, torch.ones_like(z_max), z_max)
    x_min = torch.where(x_min == 1e8, torch.zeros_like(x_min), x_min)
    y_min = torch.where(y_min == 1e8, torch.zeros_like(y_min), y_min)
    z_min = torch.where(z_min == 1e8, torch.zeros_like(z_min), z_min)

    # 堆叠为 [N, 6] 张量 (x_min, y_min, z_min, x_max, y_max, z_max)
    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_noise(boxes, box_noise_scale=0):

    known_bbox_expand = box_xyxy_to_cxcywh(boxes)

    diff = torch.zeros_like(known_bbox_expand)
    diff[:, :2] = known_bbox_expand[:, 2:] / 2
    diff[:, 2:] = known_bbox_expand[:, 2:]
    known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),diff).cuda() * box_noise_scale
    boxes = box_cxcywh_to_xyxy(known_bbox_expand)
    boxes = boxes.clamp(min=0.0, max=1024)

    return boxes

def masks_sample_points(masks,k=10):
    """Sample points on mask
    """
    if masks.numel() == 0:
        return torch.zeros((0, k, 3), device=masks.device)

        # Ensure input is 4D: [N, D, H, W]
    if masks.dim() != 4:
        raise ValueError(f"Expected 4D input [N, D, H, W], got shape {masks.shape}")

    n, d, h, w = masks.shape

    # Create coordinate grids for depth, height, width
    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    z, y, x = torch.meshgrid(z, y, x, indexing='ij')

    samples = []
    for b_i in range(n):
        # Binarize mask (threshold at > 0 for medical imaging masks)
        select_mask = masks[b_i] > 0
        if select_mask.sum() < k:
            raise ValueError(f"Mask {b_i} has fewer than {k} non-zero points")

        # Sample k points
        x_idx = torch.masked_select(x, select_mask)
        y_idx = torch.masked_select(y, select_mask)
        z_idx = torch.masked_select(z, select_mask)

        perm = torch.randperm(x_idx.size(0))[:k]
        samples_x = x_idx[perm]
        samples_y = y_idx[perm]
        samples_z = z_idx[perm]
        samples_xyz = torch.stack([samples_x, samples_y, samples_z], dim=-1)
        samples.append(samples_xyz)

    return torch.stack(samples)


# Add noise to mask input
# From Mask Transfiner https://github.com/SysCV/transfiner
def masks_noise(masks):
    def get_incoherent_mask(input_masks, sfact):
        mask = input_masks.float()
        d, h, w = input_masks.shape[-3:]
        # Downsample to (D//sfact, H//sfact, W//sfact)
        mask_small = F.interpolate(mask, size=(d // sfact, h // sfact, w // sfact), mode='nearest')
        # Upsample back to (D, H, W)
        mask_recover = F.interpolate(mask_small, size=(d, h, w), mode='nearest')
        mask_residue = (mask - mask_recover).abs()
        mask_residue = (mask_residue >= 0.01).float()
        return mask_residue
    # Assume binary masks (0 or 1); no normalization needed
    mask_binary = (masks > 0).float()
    # Generate noise
    mask_noise = torch.randn_like(mask_binary) * 1.0
    # Apply incoherent mask
    inc_masks = get_incoherent_mask(mask_binary, 8)
    # Add noise and threshold to create binary output
    noisy_masks = ((mask_binary + mask_noise * inc_masks) > 0.5).float()

    return noisy_masks


def mask_iou(pred_label,label):
    '''
    calculate mask iou for pred_label and gt_label
    '''

    pred_label = (pred_label>0)[0].int()
    label = (label>128)[0].int()

    intersection = ((label * pred_label) > 0).sum()
    union = ((label + pred_label) > 0).sum()
    return intersection / union



# General util function to get the boundary of a binary mask.
# https://gist.github.com/bowenc0221/71f7a02afee92646ca05efeeb14d687d
def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask for 2D or 3D masks.
    Args:
        mask: numpy array, shape [H, W] or [D, H, W], binary mask
        dilation_ratio: float, ratio to calculate dilation
    Returns:
        boundary mask (numpy array)
    """
    if mask.ndim == 3:  # 3D mask [D, H, W]
        d, h, w = mask.shape
        img_diag = np.sqrt(h ** 2 + w ** 2)  # Use H and W for dilation
        dilation = max(1, int(round(dilation_ratio * img_diag)))

        # Pad mask
        new_mask = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
        kernel = np.ones((3, 3), dtype=np.uint8)  # 2D kernel for H, W
        new_mask_erode = new_mask.copy()
        for z in range(d):
            new_mask_erode[z] = cv2.erode(new_mask[z], kernel, iterations=dilation)
        mask_erode = new_mask_erode[:, 1:h + 1, 1:w + 1]
        return mask - mask_erode
    elif mask.ndim == 2:  # 2D mask [H, W]
        h, w = mask.shape
        img_diag = np.sqrt(h ** 2 + w ** 2)
        dilation = max(1, int(round(dilation_ratio * img_diag)))
        new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
        mask_erode = new_mask_erode[1:h + 1, 1:w + 1]
        return mask - mask_erode
    else:
        raise ValueError(f"Expected 2D or 3D mask, got shape {mask.shape}")


def boundary_iou(gt, dt, dilation_ratio=0.02):
    """
    Compute boundary IoU between two binary masks (2D or 3D).
    Args:
        gt: [1, D, H, W] or [1, H, W], torch tensor
        dt: [C, D, H, W] or [C, H, W], torch tensor
        dilation_ratio: float, ratio to calculate dilation
    Returns:
        boundary IoU (float tensor)
    """
    device = gt.device
    gt = (gt > 128).float()  # [1, D, H, W] or [1, H, W]
    dt = (dt > 0).float()  # [C, D, H, W] or [C, H, W]

    # 扩展 gt 的通道维度以匹配 dt
    if gt.shape[0] != dt.shape[0]:
        gt = gt.expand(dt.shape[0], *gt.shape[1:])  # [1, ...] -> [C, ...]

    # 处理 3D 或 2D 输入
    if gt.dim() == 4:  # 3D: [C, D, H, W]
        gt_np = gt.cpu().numpy()  # [C, D, H, W]
        dt_np = dt.cpu().numpy()  # [C, D, H, W]
        intersection = 0
        union = 0
        for c in range(gt_np.shape[0]):
            gt_bound = mask_to_boundary(gt_np[c], dilation_ratio)  # [D, H, W]
            dt_bound = mask_to_boundary(dt_np[c], dilation_ratio)  # [D, H, W]
            intersection += ((gt_bound * dt_bound) > 0).sum()
            union += ((gt_bound + dt_bound) > 0).sum()
        iou = intersection / union if union > 0 else 0
    else:  # 2D: [C, H, W]
        gt_np = gt.cpu().numpy()  # [C, H, W]
        dt_np = dt.cpu().numpy()  # [C, H, W]
        intersection = 0
        union = 0
        for c in range(gt_np.shape[0]):
            gt_bound = mask_to_boundary(gt_np[c], dilation_ratio)  # [H, W]
            dt_bound = mask_to_boundary(dt_np[c], dilation_ratio)  # [H, W]
            intersection += ((gt_bound * dt_bound) > 0).sum()
            union += ((gt_bound + dt_bound) > 0).sum()
        iou = intersection / union if union > 0 else 0

    return torch.tensor(iou, dtype=torch.float, device=device)