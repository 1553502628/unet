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

# ==========================================
# 1. 导入你的核心模型
# ==========================================
from src.module.unet import UNet
from src.SURMT.Networks import SURMT  # 包含了 HSAModule (DWT+WFCA+DCT) 瓶颈层的网络

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
    大小器官尺度解耦双分支包装器 (Scale-Decoupled Dual-Head Wrapper)
    目标小器官: [2:胆囊, 6:胰腺] (左右肾已回归大器官组)
    """

    def __init__(self, base_model: nn.Module, num_classes: int = 9, small_organ_ids=(2, 6)):
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
            target_conv.register_forward_pre_hook(self._save_feat_hook)
        else:
            feat_channels = num_classes

        # 构建小器官专属细化头 (3分类：0 非小器官，1 胆囊，2 胰腺)
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

        self.fusion_alpha = nn.Parameter(torch.tensor(0.5))

    def _save_feat_hook(self, module, inputs):
        self._decoupled_feat = inputs[0]

    def forward(self, x: torch.Tensor):
        self._decoupled_feat = None
        base_out = self.base_model(x)
        main_logits = base_out['out'] if isinstance(base_out, dict) else base_out

        feat = self._decoupled_feat if self._decoupled_feat is not None else main_logits
        small_logits = self.small_organ_head(feat)

        if small_logits.shape[-2:] != main_logits.shape[-2:]:
            small_logits = nn.functional.interpolate(
                small_logits, size=main_logits.shape[-2:], mode='bilinear', align_corners=False
            )

        fused_logits = main_logits.clone()
        small_fg_delta = small_logits[:, 1:] - small_logits[:, 0:1]
        alpha = torch.clamp(self.fusion_alpha, 0.1, 2.0)

        for idx, organ_id in enumerate(self.small_organ_ids):
            if organ_id < self.num_classes:
                fused_logits[:, organ_id] = fused_logits[:, organ_id] + alpha * small_fg_delta[:, idx]

        return {'out': fused_logits, 'small_out': small_logits}


class SynapseTransform:
    """
    严格禁止左右翻转（保护肾脏定位）
    采用随机仿射变换（多尺度缩放 + 小角度旋转）+ 强度扰动
    """

    def __init__(self, train: bool):
        self.train = train

    def __call__(self, image, target):
        if self.train:
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

            if random.random() > 0.5:
                contrast_scale = random.uniform(0.9, 1.1)
                brightness_shift = random.uniform(-0.05, 0.05)
                image = torch.clamp(image * contrast_scale + brightness_shift, 0.0, 1.0)

        return image, target


def get_transform(train):
    return SynapseTransform(train)


def create_model(num_classes):
    # ==========================================
    # 2. 核心修改：真正接入带有高级瓶颈层的网络
    # ==========================================
    # 这里启用 SURMT，它内部集成了你需要的 HSAModule（频域瓶颈层处理）
    # 如果你想用自己魔改的 UNet16/UNet22，在此处替换即可，例如：
    # base_model = UNet22(in_channels=1, num_classes=num_classes)
    base_model = SURMT(channels=1, classes=num_classes)

    # 开启方案 B：自动挂载胆囊(2)和胰腺(6)专属的小器官细化分支
    if num_classes == 9:
        model = ScaleDecoupledWrapper(base_model, num_classes=num_classes, small_organ_ids=(2, 6))
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

    # ==========================================
    # 3. 核心修改：修复 Windows 下 DataLoader 慢的问题
    # ==========================================
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

    # 打印网络结构核实瓶颈层是否接入
    print("=" * 50)
    print(
        f"当前启用的底层主干模型: {model.base_model.__class__.__name__ if hasattr(model, 'base_model') else model.__class__.__name__}")
    print("=" * 50)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # 使用 AdamW，比 SGD 更有利于带有注意力和频域模块的网络收敛
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr, weight_decay=args.weight_decay
    )

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # warmup_epochs=5 即可，给后面的下降留出空间
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=5)

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

    # ==========================================
    # 4. 核心修改：缩减 epoch 到 110
    # ==========================================
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