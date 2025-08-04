import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import matplotlib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from segment_anything_3d import sam_model_registry3D
from segment_anything_3d.modeling import TwoWayTransformer, MaskDecoder3D, TwoWayTransformer3D
import torch.nn.functional as F
from utils.click_method import get_next_click3D_torch_2

matplotlib.use('Agg')
from torch.utils.data._utils.collate import default_collate

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

def dice_coefficient(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    return (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)


class NiftiDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_size=(16, 128, 128), image_files=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_size = target_size
        if image_files is None:
            self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        else:
            self.images = sorted(image_files)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.images[idx])
        img_nii = nib.load(img_path)
        label_nii = nib.load(label_path)
        img = img_nii.get_fdata()
        label = label_nii.get_fdata()
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)
        label = label.astype(np.uint8)
        current_shape = img.shape
        zoom_factors = [self.target_size[i] / current_shape[i] for i in range(3)]
        img = zoom(img, zoom_factors, order=3)
        label = zoom(label, zoom_factors, order=0)
        if img.shape != self.target_size:
            raise ValueError(f"Resized shape {img.shape} does not match target_size {self.target_size}")
        unique_labels = np.unique(label)
        img = torch.from_numpy(img).float().unsqueeze(0)
        label = torch.from_numpy(label).long()
        if self.transform:
            img, label = self.transform(img, label)
        return img, label, unique_labels, img_path


class UNet3D(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet3D, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottom = conv_block(512, 1024)
        self.upconv4 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec1 = conv_block(128, 64)
        self.final_conv = nn.Conv3d(64, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottom(self.pool(e4))
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.final_conv(d1)


def custom_collate_fn(batch):
    imgs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    unique_labels = [item[2] for item in batch]
    img_paths = [item[3] for item in batch]
    imgs = default_collate(imgs)
    labels = default_collate(labels)
    return imgs, labels, unique_labels, img_paths

def plot_and_save_curves(all_avg_dice_scores, all_avg_iou_scores, epoch, output_dir):
    """Plots and saves average Dice and IoU curves."""
    epochs_evaluated = list(range(2, len(all_avg_dice_scores) * 2 + 1, 2))

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Plot average Dice scores
    axes[0].plot(epochs_evaluated, all_avg_dice_scores, linestyle='-', color='blue', label='Average Dice')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Dice')
    axes[0].set_title('Validation Average Dice Curve')
    axes[0].grid(True)
    axes[0].legend()

    # Plot average IoU scores
    axes[1].plot(epochs_evaluated, all_avg_iou_scores, linestyle='-', color='green', label='Average IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Average IoU')
    axes[1].set_title('Validation Average IoU Curve')
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    curves_path = os.path.join(output_dir, f'validation_curves_epoch_{epoch+1}.png')
    plt.savefig(curves_path)
    plt.close()
    print(f'Validation curves saved to {curves_path}')

click_methods = {
    'random': get_next_click3D_torch_2,
}
def get_points(prev_masks, gt3D,device,click_points, click_labels):
    multi_click = False
    batch_points, batch_labels = click_methods['random'](prev_masks, gt3D)

    points_co = torch.cat(batch_points, dim=0).to(device)
    points_la = torch.cat(batch_labels, dim=0).to(device)

    click_points.append(points_co)
    click_labels.append(points_la)

    points_multi = torch.cat(click_points, dim=1).to(device)
    labels_multi = torch.cat(click_labels, dim=1).to(device)

    if multi_click:
        points_input = points_multi
        labels_input = labels_multi
    else:
        points_input = points_co
        labels_input = points_la
    return points_input, labels_input

def batch_forward(sam_model, image_embedding, gt3D, low_res_masks, points=None):

    sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
        points=points,
        boxes=None,
        masks=low_res_masks,
    )
    low_res_masks, iou_predictions = sam_model.mask_decoder(
        image_embeddings=image_embedding,  # (B, 256, 64, 64)
        image_pe=sam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    prev_masks = F.interpolate(low_res_masks,
                               size=gt3D.shape[-3:],
                               mode='trilinear',
                               align_corners=False)
    conv = nn.Conv3d(prev_masks.shape[1], 20, kernel_size=1).to(low_res_masks.device)
    prev_masks = conv(prev_masks)
    return low_res_masks, prev_masks

def interaction(sam_model, image_embedding, gt3D, num_clicks):
    return_loss = 0
    click_points = []
    click_labels = []
    prev_masks = torch.zeros_like(gt3D).to(gt3D.device)
    low_res_masks = F.interpolate(prev_masks.float(), size=(32, 32))
    random_insert = np.random.randint(2, 9)
    for num_click in range(num_clicks):
        points_input, labels_input = get_points(prev_masks, gt3D, gt3D.device, click_points, click_labels)

        if num_click == random_insert or num_click == num_clicks - 1:
            low_res_masks, prev_masks = batch_forward(sam_model,
                                                       image_embedding,
                                                       gt3D,
                                                       low_res_masks,
                                                       points=None)
        else:
            low_res_masks, prev_masks = batch_forward(sam_model,
                                                       image_embedding,
                                                       gt3D,
                                                       low_res_masks,
                                                       points=[points_input, labels_input])

    return prev_masks

def train_model(model, sam, train_loader, test_loader, optimizer, criterion, device, num_epochs, n_classes=20, output_dir='output'):
    model.to(device)
    model.train()
    epoch_losses = []
    epoch_dice_scores = []
    epoch_iou_scores = []

    all_avg_dice_scores = []
    all_avg_iou_scores = []

    best_loss = float('inf')
    sam = sam.module if isinstance(sam, nn.DataParallel) else sam
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')
        for inputs, labels, unique_labels_batch, _ in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)

            fused_input = torch.cat([inputs, outputs], dim=1)  # 形状 [1, 21, 16, 128, 128]
            # 如果需要 1 通道，降维（例如取均值或第一个通道）
            if fused_input.shape[1] > 1:
                fused_input = fused_input.mean(dim=1, keepdim=True)  # 形状 [1, 1, 16, 128, 128]
            image_embedding = sam.image_encoder(fused_input)
            # image_embedding = sam.image_encoder(inputs)
            prev_masks = interaction(sam, image_embedding, labels, num_clicks=5)

            outputs_drsam = prev_masks + outputs

            batch_size = inputs.shape[0]
            loss = 0
            for b in range(batch_size):
                mask = torch.zeros(n_classes, device=device)
                for lbl in unique_labels_batch[b]:
                    mask[int(lbl)] = 1
                masked_output = outputs_drsam[b] * mask.view(-1, 1, 1, 1)
                loss += criterion(masked_output.unsqueeze(0), labels[b:b + 1])
            loss = loss / batch_size
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            del inputs, labels, outputs, outputs_drsam, loss
            torch.cuda.empty_cache()

        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

        # Save latest model weights
        os.makedirs(output_dir, exist_ok=True)
        latest_model_path = os.path.join(output_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, latest_model_path)
        print(f'Saved latest model: {latest_model_path}')

        # Save best model based on loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f'Saved best model: {best_model_path}')

        # Evaluate and save every 10 epochs
        if epoch % 2 == 0:
            dice_scores, iou_scores =  evaluate_and_save(model, test_loader, device, n_classes, output_dir, epoch, all_avg_dice_scores, all_avg_iou_scores)
            epoch_dice_scores.append(dice_scores)
            epoch_iou_scores.append(iou_scores)
        else:
            epoch_dice_scores.append(None)
            epoch_iou_scores.append(None)

    # Plot and save loss curve (only loss curve at the end)
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(7, 7))
    plt.plot(epochs_range, epoch_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)

    loss_plot_path = os.path.join(output_dir, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.close()
    print(f'Loss curve saved to {loss_plot_path}')

    return epoch_losses
def draw_seg_nii(outputs, b, img_paths, device, n_classes, output_dir):
    prob = outputs[b].softmax(dim=0)  # 形状: [C, D, H, W]
    orig_nii = nib.load(img_paths[b])
    orig_shape = orig_nii.shape[:3]  # 确保 3 维: [D', H', W']
    affine = orig_nii.affine
    header = orig_nii.header.copy()

    # GPU 插值
    prob_torch = prob.unsqueeze(0)  # [1, C, D, H, W]
    prob_resized = F.interpolate(prob_torch, size=orig_shape, mode='trilinear', align_corners=False).squeeze(
        0)  # [C, D', H', W']

    if device.type == 'cuda':
        kernel = torch.ones(n_classes, 1, 3, 3, 3, device=device) / 27  # [C, 1, 3, 3, 3]
        prob_resized = F.conv3d(
            prob_resized.unsqueeze(0),  # [1, C, D', H', W']
            kernel,
            padding=1,
            groups=n_classes  # 每个通道独立卷积
        ).squeeze(0)  # [C, D', H', W']

    # CPU 处理
    prob_resized = prob_resized.cpu().numpy()  # [C, D', H', W']
    pred_resized = np.argmax(prob_resized, axis=0).astype(np.uint8)

    # 设置 NIfTI 头信息
    header.set_data_dtype(np.uint8)
    header.set_zooms(orig_nii.header.get_zooms())
    output_path = os.path.join(output_dir, os.path.basename(img_paths[b]))
    nii_img = nib.Nifti1Image(pred_resized, affine, header)
    nib.save(nii_img, output_path)

    try:
        saved_nii = nib.load(output_path)
        print(f"Verified {output_path}: shape = {saved_nii.shape}, dtype = {saved_nii.get_fdata().dtype}")
    except Exception as e:
        print(f"Error verifying {output_path}: {e}")

def evaluate_and_save(model, test_loader, device, n_classes, output_dir,epoch, all_avg_dice_scores, all_avg_iou_scores):
    model.eval()
    dice_scores = {f'class_{i}': [] for i in range(n_classes)}
    iou_scores = {f'class_{i}': [] for i in range(n_classes)}

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc='Evaluating', unit='batch')
        for inputs, labels, unique_labels_batch, img_paths in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            for b in range(inputs.shape[0]):
                pred = preds[b].cpu().numpy()
                label = labels[b].cpu().numpy()
                unique_labels = unique_labels_batch[b]
                if draw_seg and epoch % 20 == 0:
                    draw_seg_nii(outputs, b, img_paths, device, n_classes, output_dir)


                for c in unique_labels:
                    pred_c = (pred == c).astype(np.float32)
                    label_c = (label == c).astype(np.float32)
                    dice = dice_coefficient(torch.tensor(pred_c), torch.tensor(label_c)).item()
                    iou = jaccard_score(label_c.flatten(), pred_c.flatten())
                    dice_scores[f'class_{c}'].append(dice)
                    iou_scores[f'class_{c}'].append(iou)

    avg_dice_scores = {}
    avg_iou_scores = {}

    total_dice = 0.0
    total_iou = 0.0
    valid_classes = 0

    for c in range(n_classes):
        if dice_scores[f'class_{c}']:
            avg_dice = np.mean(dice_scores[f'class_{c}'])
            avg_iou = np.mean(iou_scores[f'class_{c}'])
            total_dice += avg_dice
            total_iou += avg_iou
            valid_classes += 1
            avg_dice_scores[f'class_{c}'] = avg_dice
            avg_iou_scores[f'class_{c}'] = avg_iou
            print(f'Class {c}: Average Dice: {avg_dice:.4f}, Average IoU: {avg_iou:.4f}')
        else:
            avg_dice_scores[f'class_{c}'] = 0.0
            avg_iou_scores[f'class_{c}'] = 0.0
            print(f'Class {c}: No instances found in test set')

    if valid_classes > 0:
        avg_dice = total_dice / valid_classes
        avg_iou = total_iou / valid_classes
        # print(f'Epoch {epoch}: Average Dice (all classes): {avg_dice:.4f}, Average IoU (all classes): {avg_iou:.4f}')
    else:
        avg_dice = 0.0
        avg_iou = 0.0
        # print(f'Epoch {epoch}: No instances found in test set for any class')

    # Store current epoch's average scores
    all_avg_dice_scores.append(avg_dice)
    all_avg_iou_scores.append(avg_iou)
    # print(all_avg_dice_scores, all_avg_iou_scores)
    # Plotting and saving curves
    plot_and_save_curves(all_avg_dice_scores, all_avg_iou_scores, epoch, output_dir)

    return all_avg_dice_scores, all_avg_iou_scores

def init_checkpoint(model, optimizer, lr_scheduler, ckp_path, device):
    last_ckpt = None
    if os.path.exists(ckp_path):
        last_ckpt = torch.load(ckp_path, map_location=device, weights_only=False)

    if last_ckpt:
        model.load_state_dict(last_ckpt['model_state_dict'])
        start_epoch = last_ckpt['epoch']
        optimizer.load_state_dict(last_ckpt['optimizer_state_dict'])
        lr_scheduler.load_state_dict(last_ckpt['lr_scheduler_state_dict'])
        losses = last_ckpt['losses']
        dices = last_ckpt['dices']
        best_loss = last_ckpt['best_loss']
        best_dice = last_ckpt['best_dice']
        print(f"Loaded checkpoint from {ckp_path} (epoch {start_epoch})")
    else:
        start_epoch = 0
        print(f"No checkpoint found at {ckp_path}, start training from scratch")

def main():
    pretrain = True
    global draw_seg
    draw_seg = True
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    image_dir = '/Data/VesselFM/dataset/imagesTr'
    label_dir = '/Data/VesselFM/dataset/labelsTr'
    output_dir = '/Data/VesselFM/3D_DrSAM_unet/output_filter'
    checkpoint = '/Data/VesselFM/FastSAM3D/ckpt/sam_med3d.pth'
    n_classes = 20
    batch_size = 1
    num_epochs = 3000
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    train_files, test_files = train_test_split(image_files, test_size=0.1, random_state=42)
    print(f"Training files: {len(train_files)}, Testing files: {len(test_files)}")

    train_dataset = NiftiDataset(image_dir, label_dir, target_size=(16, 128, 128), image_files=train_files)
    test_dataset = NiftiDataset(image_dir, label_dir, target_size=(16, 128, 128), image_files=test_files)

    train_loader = DataLoader(train_dataset, batch_size=batch_size * torch.cuda.device_count(), shuffle=True,
                              collate_fn=custom_collate_fn, num_workers=4 * torch.cuda.device_count())
    test_loader = DataLoader(test_dataset, batch_size=batch_size * torch.cuda.device_count(), shuffle=False,
                             collate_fn=custom_collate_fn, num_workers=4 * torch.cuda.device_count())

    model = UNet3D(in_channels=1, n_classes=n_classes)
    sam = sam_model_registry3D['vit_b'](checkpoint=None)


    print("--- define optimizer ---")

    # optimizer_sam = optim.AdamW(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-02,
    #                             weight_decay=0.0001)
    optimizer_sam = torch.optim.AdamW(
        [
            {
                'params': sam.image_encoder.parameters()
            },  # , 'lr': self.args.lr * 0.1},
            {
                'params': sam.prompt_encoder.parameters(),
                'lr': learning_rate * 0.1
            },
            {
                'params': sam.mask_decoder.parameters(),
                'lr': learning_rate * 0.1
            },
        ],
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0001)
    lr_scheduler_sam = torch.optim.lr_scheduler.StepLR(optimizer_sam, 3)
    lr_scheduler_sam.last_epoch = 0

    init_checkpoint(model=sam, optimizer=optimizer_sam, lr_scheduler=lr_scheduler_sam, ckp_path=checkpoint,
                    device=device)

    sam = sam.to(device=device)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Check for existing weights
    checkpoint_path = os.path.join(output_dir, 'latest_model.pth')
    start_epoch = 0
    if os.path.exists(checkpoint_path) and pretrain:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']

        # 检查 state_dict 键是否以 module. 开头，并与模型需求匹配
        model_keys = model.state_dict().keys()
        state_dict_has_module = any(k.startswith('module.') for k in state_dict.keys())
        model_expects_module = any(k.startswith('module.') for k in model_keys)

        # 调整 state_dict 键
        new_state_dict = {}
        if state_dict_has_module and not model_expects_module:
            # 移除 module. 前缀
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not state_dict_has_module and model_expects_module:
            # 添加 module. 前缀
            new_state_dict = {f"module.{k}": v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict  # 无需调整

        # 加载 state_dict，忽略不匹配的键（如 num_batches_tracked）
        try:
            model.load_state_dict(new_state_dict, strict=False)
        except RuntimeError as e:
            print(f"Error loading state_dict: {e}")
            raise

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        model.to(device)

        # 确保优化器状态张量在正确设备上
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        print(f"Loaded checkpoint from {checkpoint_path}, resuming from epoch {start_epoch}")
    else:
        model.to(device)
        print("No checkpoint found, starting training from scratch")


    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        sam = nn.DataParallel(sam)
    # Train model
    train_model(model, sam, train_loader,test_loader, optimizer, criterion, device, num_epochs - start_epoch, n_classes, output_dir)


    # Evaluate and save results
    # dice_scores, iou_scores = evaluate_and_save(model, test_loader, device, n_classes, output_dir, , all_avg_dice_scores, all_avg_iou_scores)


if __name__ == '__main__':
    main()

