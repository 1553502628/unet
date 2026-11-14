import os
import time
import h5py
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

# ==========================================
# 1. 导入你完整的包含了 UNet 和自定义模块的模型类
# (请确保这里的类名与你 train.py 中训练时使用的完整类一致)
# ==========================================
from src.SURMT.Networks import SURMT

# 如果你的类名叫其他名字，请在这里 import 并修改下方的实例化代码


# ==========================================
# 2. 定义 Synapse 9类别调色板 (RGB格式)
# 0背景, 1主动脉, 2胆囊, 3左肾, 4右肾, 5肝脏, 6胰腺, 7脾脏, 8胃
# ==========================================
SYNAPSE_PALETTE = np.array([
    [0, 0, 0],  # 0: Background (黑)
    [255, 128, 0],  # 1: Aorta 主动脉 (橙)
    [255, 255, 0],  # 2: Gallbladder 胆囊 (黄)
    [0, 0, 255],  # 3: Kidney(L) 左肾 (蓝)
    [0, 255, 0],  # 4: Kidney(R) 右肾 (绿)
    [0, 255, 255],  # 5: Liver 肝脏 (青)
    [128, 0, 255],  # 6: Pancreas 胰腺 (深紫)
    [255, 0, 0],  # 7: Spleen 脾脏 (红)
    [255, 0, 255]  # 8: Stomach 胃 (品红)
], dtype=np.uint8)


def decode_segmap(mask_index, num_classes=9):
    """将单通道的类别索引图(0-8)转换为3通道的RGB彩色图"""
    rgb_image = np.zeros((mask_index.shape[0], mask_index.shape[1], 3), dtype=np.uint8)
    for i in range(1, num_classes):  # 跳过背景(0)
        rgb_image[mask_index == i] = SYNAPSE_PALETTE[i]
    return rgb_image


def main():
    num_classes = 9
    weights_path = "save_weights/par9/best_model.pth"

    # 读取 H5 文件的目录
    h5_load_dir = './data/Synapse/test_vol_h5'
    save_path = './prediction/synapse_results'

    os.makedirs(save_path, exist_ok=True)
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(h5_load_dir), f"Directory {h5_load_dir} not found."

    h5_file_list = [f for f in os.listdir(h5_load_dir) if f.endswith('.h5') or f.endswith('.hdf5')]

    # 单通道归一化参数
    mean = [0.709]
    std = [0.127]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ==========================================
    # 3. 创建完整的自定义模型并加载全套权重
    # ==========================================
    # 必须使用你结合了 UNet 和自有模块的完整类，千万不要只实例化 UNet
    model = SURMT(channels=1, classes=num_classes)

    # 直接加载完整的权重字典，包含了 base_model、小器官头和融合参数
    checkpoint = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)

    model.to(device)
    model.eval()

    print(f"Start predicting {len(h5_file_list)} volume files...")

    # ==========================================
    # 4. 遍历 H5 文件与切片，进行推理和渲染
    # ==========================================
    for h5_name in h5_file_list:
        h5_path = os.path.join(h5_load_dir, h5_name)
        vol_name = h5_name.split('.')[0]

        # 读取 H5 体数据 (Synapse shape 通常为: 切片数 x H x W)
        h5f = h5py.File(h5_path, 'r')
        image_vol = h5f['image'][:]

        print(f"Processing Volume: {vol_name} | Slices: {image_vol.shape[0]}")

        for slice_idx in range(image_vol.shape[0]):
            slice_np = image_vol[slice_idx, :, :]  # 取出 2D 切片 shape: (H, W)

            # --- 构造用于可视化叠加的 RGB 原图底片 ---
            slice_vis = ((slice_np - slice_np.min()) / (slice_np.max() - slice_np.min() + 1e-8) * 255).astype(np.uint8)
            original_img = Image.fromarray(slice_vis).convert('RGB')

            # --- 构造用于模型输入的 Tensor ---
            img_tensor = torch.from_numpy(slice_np).float().unsqueeze(0)  # (1, H, W)

            # 归一化 (如果你的 H5 数据加载出来已经是 0~1 之间，请注释掉下面这行)
            img_tensor = transforms.Normalize(mean=mean, std=std)(img_tensor)

            img = torch.unsqueeze(img_tensor, dim=0)  # (1, 1, H, W)
            img_height, img_width = img.shape[-2:]

            # Padding 到 32 的整数倍 (避免 UNet 下采样和上采样拼接时尺寸不匹配)
            stride = 32
            pad_h = (stride - img_height % stride) % stride
            pad_w = (stride - img_width % stride) % stride
            if pad_h > 0 or pad_w > 0:
                img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')

            with torch.no_grad():
                output = model(img.to(device))

                # 兼容你自定义模块可能返回多个分支结果的情况
                if isinstance(output, dict):
                    logits = output['out']
                elif isinstance(output, tuple) or isinstance(output, list):
                    logits = output[0]  # 取主分支输出 (通常放在第0个)
                else:
                    logits = output

                # 去除 Padding 带来的额外预测区域
                if pad_h > 0 or pad_w > 0:
                    logits = logits[..., :img_height, :img_width]

                # 获取概率最大的类别索引 shape: (H, W)
                prediction = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

                # ==========================================
                # 5. 生成彩色 Mask 并与原图半透明叠加
                # ==========================================
                # 如果这一层切片里确实没有任何器官(预测全为0)，直接保存原图
                if np.max(prediction) == 0:
                    final_img = original_img
                else:
                    # 将类别索引图映射为 3 通道的彩色 Mask
                    color_mask_np = decode_segmap(prediction, num_classes=num_classes)
                    color_mask_img = Image.fromarray(color_mask_np).convert('RGB')

                    # 半透明融合 (alpha=0.4 调整颜色深度)
                    blended_img = Image.blend(original_img, color_mask_img, alpha=0.4)

                    orig_np = np.array(original_img)
                    blend_np = np.array(blended_img)

                    # 严谨的背景剔除：基于 prediction 索引，把背景处变回纯黑白原图
                    mask_bg = (prediction == 0)[..., None]
                    final_np = np.where(mask_bg, orig_np, blend_np)
                    final_img = Image.fromarray(final_np.astype(np.uint8))

                # 保存最终渲染好的叠加图
                save_filename = f"overlay_{vol_name}_slice{slice_idx:03d}.png"
                final_img.save(os.path.join(save_path, save_filename))

        h5f.close()


if __name__ == '__main__':
    main()