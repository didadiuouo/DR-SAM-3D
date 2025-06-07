# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
# from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import nibabel as nib
import torch.nn as nn


#### --------------------- dataloader online ---------------------####

def get_im_gt_name_dict(datasets, flag='valid'):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        tmp_im_list, tmp_gt_list = [], []
        tmp_im_list = glob(datasets[i]["im_dir"]+os.sep+'*'+datasets[i]["im_ext"])
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))

        if(datasets[i]["gt_dir"]==""):
            print('-gt-', datasets[i]["name"], datasets[i]["gt_dir"], ': ', 'No Ground Truth Found')
            tmp_gt_list = []
        else:
            tmp_gt_list = [datasets[i]["gt_dir"]+os.sep+x.split(os.sep)[-1].split(datasets[i]["im_ext"])[0]+datasets[i]["gt_ext"] for x in tmp_im_list]
            print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"]})

    return name_im_gt_list

def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, training=False):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 1
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8

    for i in range(len(name_im_gt_list)):
        if training:
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform=transforms.Compose(my_transforms))
            gos_datasets.append(gos_dataset)
        else:
            gos_dataset = OnlineDataset([name_im_gt_list[i]], transform=transforms.Compose(my_transforms),
                                        eval_ori_resolution=True)
            dataloader = DataLoader(gos_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_)
            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    if training:
        gos_dataset = ConcatDataset(gos_datasets)
        dataloader = DataLoader(gos_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_)
        gos_dataloaders = dataloader
        gos_datasets = gos_dataset
    # if training:
    #     for i in range(len(name_im_gt_list)):
    #         gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
    #         gos_datasets.append(gos_dataset)
    #
    #     gos_dataset = ConcatDataset(gos_datasets)
    #     sampler = DistributedSampler(gos_dataset)
    #     batch_sampler_train = torch.utils.data.BatchSampler(
    #         sampler, batch_size, drop_last=True)
    #     dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)
    #
    #     gos_dataloaders = dataloader
    #     gos_datasets = gos_dataset
    #
    # else:
    #     for i in range(len(name_im_gt_list)):
    #         gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
    #         sampler = DistributedSampler(gos_dataset, shuffle=False)
    #         dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)
    #
    #         gos_dataloaders.append(dataloader)
    #         gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

class RandomHFlip(object):
    def __init__(self, prob=0.5):
        """
        Initialize RandomHFlip transform for 3D images.
        Args:
            prob (float): Probability of applying horizontal flip.
        """
        self.prob = prob

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # Random horizontal flip along width (last dimension) for 3D: [C, D, H, W]
        if random.random() < self.prob:  # Note: Changed to < self.prob for consistency
            image = torch.flip(image, dims=[-1])  # Flip width (W)
            label = torch.flip(label, dims=[-1])  # Flip width (W)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}

class Resize(object):
    def __init__(self, size=(12, 880, 880)):
        """
        Initialize Resize transform for 3D images.
        Args:
            size (tuple): Desired output size (D, H, W) for 3D images.
        """
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # Ensure image and label are 5D: [C, D, H, W] -> [1, C, D, H, W]
        image = torch.unsqueeze(image, 0)  # [1, C, D, H, W]
        label = torch.unsqueeze(label, 0)  # [1, C, D, H, W]

        # Resize to 3D size
        image = F.interpolate(image, size=self.size, mode='trilinear', align_corners=False)
        label = F.interpolate(label, size=self.size, mode='nearest')

        # Squeeze back to [C, D, H, W]
        image = torch.squeeze(image, 0)
        label = torch.squeeze(label, 0)

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(self.size)}

class RandomCrop(object):
    def __init__(self, size=[128, 880, 880]):
        """
        Initialize RandomCrop transform for 3D images.
        Args:
            size (tuple): Desired crop size (D, H, W) for 3D images.
        """
        self.size = size

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        d, h, w = image.shape[1:]  # 3D dimensions: depth, height, width
        new_d, new_h, new_w = self.size

        # Random crop indices
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        depth = np.random.randint(0, d - new_d + 1)

        # Crop image and label
        image = image[:, depth:depth+new_d, top:top+new_h, left:left+new_w]
        label = label[:, depth:depth+new_d, top:top+new_h, left:left+new_w]

        return {'imidx': imidx, 'image': image, 'label': label, 'shape': torch.tensor(self.size)}
def normalize(image, mean, std):
    """Normalize a tensor image with mean and standard deviation."""
    for i in range(image.shape[0]):
        image[i] = (image[i] - mean[i]) / std[i]
    return image
class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Initialize Normalize transform for 3D images.
        Args:
            mean (list): Mean for each channel.
            std (list): Standard deviation for each channel.
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image, self.mean, self.std)
        return {'imidx': imidx, 'image': image, 'label': label, 'shape': shape}



class LargeScaleJitter(object):
    def __init__(self, scale_range=(0.5, 2.0), size=(12, 880, 880)):

        self.scale_range = scale_range
        self.size = size

    def __call__(self, sample):

        imidx, image, label, shape = sample['imidx'], sample['image'], sample['label'], sample['shape']

        # # 验证输入形状
        # if len(image.shape) != 4 or image.shape[0] != 1:
        #     raise ValueError(f"预期image形状为 [1, D, H, W]，实际为 {image.shape}")
        # if len(label.shape) != 4 or label.shape[0] != 20:
        #     raise ValueError(f"预期label形状为 [20, D, H, W]，实际为 {label.shape}")

        # 确保输入为5D张量 (N, C, D, H, W)
        if len(image.shape) == 4:
            image = image.unsqueeze(0)  # 添加批量维度: [1, 1, D, H, W]
        if len(label.shape) == 4:
            label = label.unsqueeze(0)  # 添加批量维度: [1, 20, D, H, W]

        # 随机尺度抖动，限制深度维度的缩放比例
        input_depth = image.shape[2]  # 例如 12
        scale_hw = random.uniform(self.scale_range[0], self.scale_range[1])  # H, W 的缩放比例
        scale_d = min(max(0.5, input_depth / self.size[0]), 2.0)  # 限制深度缩放，接近输入深度
        scaled_size = [
            max(1, int(self.size[0] * scale_d)),  # 深度
            max(1, int(self.size[1] * scale_hw)),  # 高度
            max(1, int(self.size[2] * scale_hw))  # 宽度
        ]

        # 调试信息
        # print(f"输入形状: {image.shape}, scaled_size: {scaled_size}, 目标形状: {self.size}")

        # 调整到中间分辨率
        image = F.interpolate(image, size=scaled_size, mode='trilinear', align_corners=False)
        label = F.interpolate(label, size=scaled_size, mode='nearest')

        # 调整到目标分辨率
        image = F.interpolate(image, size=self.size, mode='trilinear', align_corners=False)
        label = F.interpolate(label, size=self.size, mode='nearest')

        # 确保二值标签（0或1）
        label = (label > 0.5).float()  # 阈值处理，保持one-hot二值性

        # 移除批量维度
        image = image.squeeze(0)
        label = label.squeeze(0)

        return {
            'imidx': imidx,
            'image': image,
            'label': label,
            'shape': torch.tensor(self.size, dtype=torch.long)  # [128, 512, 512]
        }





class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

        self.transform = transform
        self.dataset = {}
        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
        im_ext_list = [] # im ext
        gt_ext_list = [] # gt ext
        for i in range(0,len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])


        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.eval_ori_resolution = eval_ori_resolution

    def __len__(self):
        return len(self.dataset["im_path"])
    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]

        # 用 nibabel 读取 3D 图像
        img_nii = nib.load(im_path)  # 读取 3D 图像
        im = img_nii.get_fdata()  # 获取图像数据

        # 用 nibabel 读取 3D 掩码图像
        msk_nii = nib.load(gt_path)  # 读取 3D 掩码
        gt = msk_nii.get_fdata()  # 获取掩码数据

        # Normalize image (Z-score normalization)
        im = (im - im.mean()) / (im.std() + 1e-8)  # Avoid division by zero

        # 转换为torch张量并调整图像形状为 [1, D, H, W]
        if len(im.shape) == 3:
            im = torch.tensor(im, dtype=torch.float32)  # 形状: [height, width, depth]
            im = torch.unsqueeze(im, 0)  # 添加通道维度: [1, height, width, depth]
            im = im.permute(0, 3, 1, 2)  # 调整为 [1, depth, height, width]
        elif len(im.shape) == 4 and im.shape[0] != 1:
            raise ValueError(f"预期图像具有1个通道，实际形状为 {im.shape}")

        # 处理掩码为单通道二值掩码（前景 vs. 背景）
        if len(gt.shape) == 3:
            gt = torch.tensor(gt, dtype=torch.float32)  # 形状: [depth, height, width]
            gt = (gt > 0).float()  # 前景（1-19）为 1，背景（0）为 0
            gt = torch.unsqueeze(gt, 0)  # 添加通道维度: [1, depth, height, width]
            gt = gt.permute(0, 3, 1, 2)  # 调整为 [1, depth, height, width]
        elif len(gt.shape) == 4 and gt.shape[0] == 20:
            gt = torch.tensor(gt, dtype=torch.float32)  # 形状: [20, depth, height, width]
            if not torch.all((gt == 0) | (gt == 1)):
                raise ValueError(f"预期one-hot ground_truth 为二值（0或1），实际包含非二值")
            # 合并 1-19 类为前景（忽略类别 0）
            gt = torch.sum(gt[1:], dim=0, keepdim=True)  # 形状: [1, depth, height, width]
            gt = (gt > 0).float()  # 确保二值化
            gt = gt.permute(0, 3, 1, 2)  # 调整为 [1, depth, height, width]
        else:
            raise ValueError(f"预期ground_truth形状为 [D, H, W] 或 [20, D, H, W]，实际为 {gt.shape}")

        sample = {
            "imidx": torch.tensor(idx, dtype=torch.long),
            "image": im,
            "label": gt,
            "shape": torch.tensor(im.shape[1:]),  # [depth, height, width]
        }

        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = gt.clone().type(torch.uint8)  # [20, depth, height, width]
            sample["ori_im_path"] = self.dataset["im_path"][idx]
            sample["ori_gt_path"] = self.dataset["gt_path"][idx]

        return sample