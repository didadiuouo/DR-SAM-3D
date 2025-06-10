# -*- encoding: utf-8 -*-

import medim
from segment_anything_3d import sam_model_registry3D
from utils.infer_utils import validate_paired_img_gt
from utils.metric_utils import compute_metrics_drsam, print_computed_metrics
from segment_anything_3d.modeling import TwoWayTransformer, MaskDecoder3D, TwoWayTransformer3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import os.path as osp
from glob import glob
from tqdm import tqdm
import os
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
def init_checkpoint(model, ckp_path, device):
    last_ckpt = None
    if os.path.exists(ckp_path):
        last_ckpt = torch.load(ckp_path, map_location=device, weights_only=False)

    if last_ckpt:
        model.load_state_dict(last_ckpt['model_state_dict'])
        print(f"Loaded checkpoint from {ckp_path}")
    else:
        start_epoch = 0
        print(f"No checkpoint found at {ckp_path}, start training from scratch")



if __name__ == "__main__":
    ''' 1. prepare the pre-trained model with local path or huggingface url '''
    # ckpt_path = r".\checkpoint\sam_med3d_turbo.pth"
    # or you can use a local path like:
    # model = medim.create_model("SAM-Med3D", pretrained=True, checkpoint_path=ckpt_path)
    test_data_list = [
        dict(
            img_dir=r".\data\lumbar_spine\val\img",
            gt_dir=r".\data\lumbar_spine\val\msk",
            out_dir="./data/ct_AMOS/pred_sammed3d",
            ckpt_path_sam="./ckpt/sam_model_loss_best.pth",
            ckpt_path = r'work_dirs/DrSAM_b/net/epoch_50.pth'
        ),
    ]

    for test_data in test_data_list:


        net = DrMaskDecoder("vit_b")
        sam = sam_model_registry3D["vit_b"](checkpoint=None)

        if test_data['ckpt_path']:
            print("restore model from:", test_data['ckpt_path'])
            if torch.cuda.is_available():
                net.load_state_dict(torch.load(test_data['ckpt_path']))
            else:
                net.load_state_dict(torch.load(test_data['ckpt_path'], map_location="cpu"))
        init_checkpoint(model=sam, ckp_path=test_data['ckpt_path_sam'],
                        device='cpu')

        gt_fname_list = sorted(glob(osp.join(test_data["gt_dir"], "*.nii.gz")))

        for gt_fname in tqdm(gt_fname_list):
            case_name = osp.basename(gt_fname).replace(".nii.gz", "")

            ''' 2. read and pre-proceconss your input data '''
            img_path = osp.join(test_data["img_dir"], f"{case_name}.nii.gz")
            gt_path = gt_fname
            out_path = osp.join(test_data["out_dir"], f"{case_name}.nii.gz")

            ''' 3. infer with the pre-trained SAM-Med3D model '''
            print("Validation start! plz wait for some times.")
            validate_paired_img_gt(sam, net, img_path, gt_path, out_path, num_clicks=1)
            print("Validation finish! plz check your prediction.")

            ''' 4. compute the metrics of your prediction with the ground truth '''
            metrics = compute_metrics_drsam(
                gt_value=gt_path,
                pred_value=out_path,
                metrics=['dice'],
                classes=None,
            )
            print_computed_metrics(metrics, gt_path=gt_path)
