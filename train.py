import os
import time
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from src.module.unet import UNet

from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from my_dataset import SynapseDataset


def file_check(file_name):
    temp_file_name = file_name
    i = 1
    while i:
        if os.path.exists(temp_file_name + str(i - 1)):
            i = i + 1
        else:
            return temp_file_name + str(i - 1)


class ScaleDecoupledWrapper(nn.Module):
    """
    方案 B：大小器官尺度解耦双分支包装器 (Scale-Decoupled Dual-Head Wrapper)
    自动挂载在任意分割网络外部，拦截最后一层分类卷积前的特征图，
    构建专门针对 [2:胆囊, 3:左肾, 4:右肾, 6:胰腺] 的高分辨率小器官细化分支，并在推理时自动融合。
    """
    def __init__(self, base_model: nn.Module, num_classes: int = 9, small_organ_ids=(2, 3, 4, 6)):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.small_organ_ids = list(small_organ_ids)
        self._decoupled_feat = None

        # 自动定位 base_model 中输出通道数为 num_classes 的最终分类 Conv2d 层
        target_conv = None
        for m in reversed(list(self.base_model.modules())):
            if isinstance(m, nn.Conv2d) and m.out_channels == num_classes:
                target_conv = m
                break

        if target_conv is not None:
            feat_channels = target_conv.in_channels
            # 注册 forward_pre_hook 无损截取分类头之前的倒数第二层特征图
            target_conv.register_forward_pre_hook(self._save_feat_hook)
        else:
            feat_channels = num_classes

        # 构建小器官专属细化头 (Small-Organ Refinement Head)：
        # 采用两层 3x3 卷积提取精细边缘特征，输出 5 通道 (0:非小器官, 1:胆囊, 2:左肾, 3:右肾, 4:胰腺)
        mid_c = max(feat_channels, 32)
        self.small_organ_head = nn.Sequential(
            nn.Conv2d(feat_channels, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, len(self.small_organ_ids) + 1, kernel_size=1)
        )

        # 可学习的双分支残差融合门控系数（初始为 0.5）
        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

    def _save_feat_hook(self, module, inputs):
        self._decoupled_feat = inputs[0]

    def forward(self, x: torch.Tensor):
        self._decoupled_feat = None
        base_out = self.base_model(x)
        main_logits = base_out['out'] if isinstance(base_out, dict) else base_out

        # 计算小器官专属分支输出
        feat = self._decoupled_feat if self._decoupled_feat is not None else main_logits
        small_logits = self.small_organ_head(feat)

        if small_logits.shape[-2:] != main_logits.shape[-2:]:
            small_logits = nn.functional.interpolate(
                small_logits, size=main_logits.shape[-2:], mode='bilinear', align_corners=False
            )

        # 自适应 Logits 融合：将小器官专家分支相对于背景通道(channel 0)的净响应增益叠加回主分支对应通道
        fused_logits = main_logits.clone()
        small_fg_delta = small_logits[:, 1:] - small_logits[:, 0:1]
        alpha = torch.clamp(self.fusion_alpha, 0.1, 2.0)

        for idx, organ_id in enumerate(self.small_organ_ids):
            if organ_id < self.num_classes:
                fused_logits[:, organ_id] = fused_logits[:, organ_id] + alpha * small_fg_delta[:, idx]

        return {'out': fused_logits, 'small_out': small_logits}


class SynapseTransform:
    """
    符合腹部 CT 解剖学先验的数据增强：
    严禁左右/上下翻转（防止左肾 Class 3 与右肾 Class 4 混淆、破坏器官相对方位）！
    """

    def __init__(self, train: bool):
        self.train = train

    def __call__(self, image, target):
        if self.train:
            # 1. 随机仿射变换：小角度旋转 (-15°~15°) + 多尺度缩放 (0.85~1.15) + 轻微平移
            if random.random() > 0.3:
                angle = random.uniform(-15.0, 15.0)
                scale = random.uniform(0.85, 1.15)
                _, h, w = image.shape
                max_dx = int(w * 0.08)
                max_dy = int(h * 0.08)
                translate = [random.randint(-max_dx, max_dx), random.randint(-max_dy, max_dy)]

                image = F.affine(
                    image, angle=angle, translate=translate, scale=scale, shear=0.0,
                    interpolation=InterpolationMode.BILINEAR, fill=0.0
                )
                target = F.affine(
                    target.unsqueeze(0), angle=angle, translate=translate, scale=scale, shear=0.0,
                    interpolation=InterpolationMode.NEAREST, fill=0
                ).squeeze(0)

            # 2. 随机 CT 软组织对比度与亮度微调
            if random.random() > 0.5:
                contrast_scale = random.uniform(0.9, 1.1)
                brightness_shift = random.uniform(-0.05, 0.05)
                image = torch.clamp(image * contrast_scale + brightness_shift, 0.0, 1.0)

        return image, target


def get_transform(train):
    return SynapseTransform(train)


def create_model(num_classes):
    # 基础主干网络（如需切换为 SURMT，直接改为 base_model = SURMT(channels=1, classes=num_classes) 即可）
    base_model = UNet(in_channels=1, num_classes=num_classes)
    # 启用方案 B：当类别数为 9 (Synapse 8器官+背景) 时，自动包装大小器官解耦双分支头
    if num_classes == 9:
        model = ScaleDecoupledWrapper(base_model, num_classes=num_classes, small_organ_ids=(2,6))
    else:
        model = base_model
    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes + 1

    train_dataset = SynapseDataset(args.data_path,
                                   train=True,
                                   transforms=get_transform(train=True))

    val_dataset = SynapseDataset(args.data_path,
                                 train=False,
                                 transforms=get_transform(train=False))

    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=num_workers,
                                               shuffle=True,
                                               pin_memory=True,
                                               persistent_workers=(num_workers > 0),
                                               drop_last=True,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1,
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             persistent_workers=(num_workers > 0),
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes)
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=10)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    best_dice = 0.
    start_time = time.time()
    save_par = file_check("save/par")
    os.makedirs(save_par, exist_ok=True)
    save_w = file_check("save_weights/par")
    os.makedirs(save_w, exist_ok=True)
    best_num = 0

    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_loader, device, epoch, num_classes,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)
        confmat, dice, acc = evaluate(model, val_loader, device=device, num_classes=num_classes)
        val_info = str(confmat)
        print(val_info)
        print(f"Real Global Foreground Dice: {dice:.4f} (Best so far: {max(best_dice, dice):.4f})")

        with open(save_par + "/epoch{}.txt".format(epoch), "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"real_foreground_dice: {dice:.4f}\n" \
                         f"acc: {acc:.4f}\n"
            f.write(train_info + val_info + "\n\n")

        if args.save_best is True:
            if best_dice < dice:
                best_dice = dice
            else:
                continue

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if args.save_best is True:
            torch.save(save_file, save_w + "/best_model.pth")
            best_num = epoch
        else:
            torch.save(save_file, save_w + "/model_{}.pth".format(epoch))

    os.makedirs(save_w + "/{}.txt".format(best_num), exist_ok=True)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch unet training")

    parser.add_argument("--data-path", default="./data/", help="Dataset root path")
    parser.add_argument("--num-classes", default=8, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=110, type=int, metavar="N",
                        help="number of total epochs to train")

    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save-best', default=True, type=bool, help='only save best dice weights')
    parser.add_argument("--amp", default=True, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if not os.path.exists("./save_weights"):
        os.makedirs("./save_weights", exist_ok=True)

    main(args)