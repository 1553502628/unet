import torch
from torch import nn
import torch.nn.functional as F
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target

# 定义小器官在 Synapse 原始标签中的类别索引: [2:胆囊, 3:左肾, 4:右肾, 6:胰腺]
SMALL_ORGAN_IDS = [2, 6]


def build_small_organ_target(target: torch.Tensor, ignore_index: int = 255):
    """
    在线将 9 分类全局标签映射为 5 分类小器官专属标签：
    0: 背景及大器官(原 0,1,5,7,8)
    1: 胆囊(原 2), 2: 左肾(原 3), 3: 右肾(原 4), 4: 胰腺(原 6)
    """
    small_target = torch.zeros_like(target)
    if ignore_index >= 0:
        small_target[target == ignore_index] = ignore_index
    for new_idx, orig_id in enumerate(SMALL_ORGAN_IDS, start=1):
        small_target[target == orig_id] = new_idx
    return small_target


def focal_cross_entropy(logits: torch.Tensor, target: torch.Tensor, weight=None, gamma: float = 2.0, ignore_index: int = -100):
    """像素级 Focal Cross-Entropy Loss，聚焦边界与难分小器官像素"""
    logits_f32 = logits.float()
    ce_loss = F.cross_entropy(logits_f32, target, reduction='none', ignore_index=ignore_index)
    pt = torch.exp(-ce_loss)
    focal_term = (1.0 - pt) ** gamma
    loss = focal_term * ce_loss

    valid_mask = (target != ignore_index)
    if not valid_mask.any():
        return torch.tensor(0.0, device=logits.device, dtype=torch.float32)

    if weight is not None:
        safe_target = target.clone()
        safe_target[~valid_mask] = 0
        alpha_t = weight[safe_target]
        loss = loss * alpha_t

    return loss[valid_mask].mean()


def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    # 1. 计算主分支（全类别融合输出 out）的损失
    main_logits = inputs['out']
    ce_main = focal_cross_entropy(main_logits, target, weight=loss_weight, gamma=2.0, ignore_index=ignore_index)
    if dice:
        dice_target = build_target(target, num_classes, ignore_index)
        d_main = dice_loss(main_logits, dice_target, multiclass=True, ignore_index=ignore_index)
        total_loss = 0.5 * ce_main + 1.0 * d_main
    else:
        total_loss = ce_main

    # 2. 计算小器官专精分支（small_out）的独立解耦损失
        # （在 criterion 函数内部）
        if 'small_out' in inputs:
            small_logits = inputs['small_out']
            small_target = build_small_organ_target(target, ignore_index=ignore_index)
            small_num_classes = len(SMALL_ORGAN_IDS) + 1  # 3 分类 (0:非小器官, 1:胆囊, 2:胰腺)

            # 小器官分支权重: [0:非小器官背景, 1:胆囊(极难), 2:胰腺(极难)]
            small_weight = torch.as_tensor([0.2, 5.0, 5.0], device=target.device, dtype=torch.float32)
            ce_small = focal_cross_entropy(small_logits, small_target, weight=small_weight, gamma=2.0,
                                           ignore_index=ignore_index)

        if dice:
            small_dice_target = build_target(small_target, small_num_classes, ignore_index)
            d_small = dice_loss(small_logits, small_dice_target, multiclass=True, ignore_index=ignore_index)
            small_branch_loss = 0.5 * ce_small + 1.0 * d_small
        else:
            small_branch_loss = ce_small

        # 1:1 强力注入小器官专属梯度，彻底打破大器官对梯度的垄断
        total_loss = total_loss + 1.0 * small_branch_loss

    # 3. 兼容可能存在的辅助深监督分支
    if 'aux' in inputs:
        aux_logits = inputs['aux']
        ce_aux = focal_cross_entropy(aux_logits, target, weight=loss_weight, gamma=2.0, ignore_index=ignore_index)
        total_loss = total_loss + 0.4 * ce_aux

    return total_loss


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    acc = utils.Accuracy(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)

            output = model(image)
            if isinstance(output, dict):
                output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)
            acc.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()
        acc.reduce_from_all_processes()

    # 从全局混淆矩阵计算每个类别真实的 3D 体积 Dice: Dice = 2 * IoU / (1 + IoU)
    _, _, iu = confmat.compute()
    real_dice_per_class = (2.0 * iu) / (iu + 1.0 + 1e-8)

    if num_classes > 2:
        real_mean_dice = real_dice_per_class[1:].mean().item()
        # 重新划分：左右肾 (3, 4) 回归大器官
        large_ids = [1, 3, 4, 5, 7, 8]
        small_ids = [2, 6]
        large_mean = real_dice_per_class[large_ids].mean().item() * 100
        small_mean = real_dice_per_class[small_ids].mean().item() * 100
        all_dices_str = ['{:.1f}'.format(i) for i in (real_dice_per_class * 100).tolist()]
        print(f"All Classes Real Dice (0=BG, 1~8=Organs): {all_dices_str}")
        print(f"Large Organs Mean Dice: {large_mean:.2f}% | Small Organs (2,6) Mean Dice: {small_mean:.2f}%")
        print(f"Real Global 8-Organ Mean Dice: {real_mean_dice:.4f}")
    else:
        real_mean_dice = real_dice_per_class[1].item()

    return confmat, real_mean_dice, acc.value.item()


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    elif num_classes == 9:
        # 主分支全局权重：[0:背景, 1:主动脉, 2:胆囊, 3:左肾, 4:右肾, 5:肝脏, 6:胰腺, 7:脾脏, 8:胃]
        weights = [0.2, 2.5, 8.0, 2.0, 2.0, 1.0, 8.0, 1.5, 3.0]
        loss_weight = torch.as_tensor(weights, device=device)
    else:
        weights = [0.5] + [3.0] * (num_classes - 1)
        loss_weight = torch.as_tensor(weights, device=device)

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            if not isinstance(output, dict):
                output = dict(out=output)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f, last_epoch=-1)