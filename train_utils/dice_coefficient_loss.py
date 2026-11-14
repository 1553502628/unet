import torch
import torch.nn as nn


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """build target for dice coefficient"""
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        # [N, H, W] -> [N, H, W, C]
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-5):
    """强制 float32 计算 Batch-level 体积 Dice，防止 AMP (float16) 溢出"""
    x_flat = x.reshape(-1).float()
    t_flat = target.reshape(-1).float()
    if ignore_index >= 0:
        roi_mask = torch.ne(t_flat, ignore_index)
        x_flat = x_flat[roi_mask]
        t_flat = t_flat[roi_mask]

    inter = torch.dot(x_flat, t_flat)
    sets_sum = torch.sum(x_flat) + torch.sum(t_flat)
    if sets_sum == 0:
        return torch.tensor(1.0, device=x.device, dtype=torch.float32)

    return (2.0 * inter + epsilon) / (sets_sum + epsilon)


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-5):
    """验证集评估使用的多类别前景平均 Dice"""
    dice = 0.
    valid_channels = 0
    for channel in range(x.shape[1]):
        t_c = target[:, channel, ...]
        if ignore_index >= 0:
            valid_mask = torch.ne(t_c, ignore_index)
            has_organ = torch.sum(t_c[valid_mask]) > 0
        else:
            has_organ = torch.sum(t_c) > 0

        if has_organ:
            dice += dice_coeff(x[:, channel, ...], t_c, ignore_index, epsilon)
            valid_channels += 1

    if valid_channels == 0:
        return torch.tensor(1.0, device=x.device, dtype=torch.float32)

    return dice / valid_channels


def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    """
    自适应多类别 Focal Dice Loss：
    同时支持 9 分类主分支与 5 分类小器官专精分支，自动剔除第 0 维背景通道
    """
    x = nn.functional.softmax(x.float(), dim=1)

    if not multiclass:
        return 1.0 - dice_coeff(x, target, ignore_index=ignore_index)

    # 严格剔除第 0 维背景通道
    if x.shape[1] > 1:
        x = x[:, 1:]
        target = target[:, 1:]

    total_loss = 0.0
    valid_channels = 0

    for c in range(x.shape[1]):
        x_c = x[:, c, ...]
        t_c = target[:, c, ...]

        if ignore_index >= 0:
            valid_mask = torch.ne(t_c, ignore_index)
            has_organ = torch.sum(t_c[valid_mask]) > 0
        else:
            has_organ = torch.sum(t_c) > 0

        if has_organ:
            d_c = dice_coeff(x_c, t_c, ignore_index=ignore_index)
            # 指数级焦点惩罚：Dice 越低的通道，梯度权重越大
            c_loss = (1.0 - d_c) ** 1.33
            total_loss += c_loss
            valid_channels += 1
        else:
            # 抑制空白切片上的假阳性噪声
            fp_sum = torch.sum(x_c)
            if fp_sum > 10.0:
                total_loss += 0.05 * (fp_sum / x_c.numel())

    if valid_channels == 0:
        return total_loss

    return total_loss / valid_channels