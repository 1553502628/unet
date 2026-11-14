import os
import numpy as np
import nibabel as nib
import imageio
from tqdm import tqdm


def normalize_slice(slice_2d):
    """
    将切片数据归一化到 0-255 范围
    注意：对于CT/MRI，直接最小最大值归一化可能不是最佳视觉效果，
    如有需要可在此处加入窗宽窗位调整逻辑
    """
    min_val = slice_2d.min()
    max_val = slice_2d.max()

    # 防止除以零错误（全黑图像）
    if max_val == min_val:
        return np.zeros_like(slice_2d, dtype=np.uint8)

    normalized = ((slice_2d - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return normalized


def batch_nii_to_png(input_dir, output_dir):
    """
    批量将 input_dir 下的 nii/nii.gz 文件转换为 png
    每个 nii 文件会在 output_dir 下生成一个同名子文件夹
    """
    # 创建输出根目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 nii 和 nii.gz 文件
    files = [f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]

    if not files:
        print(f"在 {input_dir} 中未找到 .nii 或 .nii.gz 文件")
        return

    print(f"找到 {len(files)} 个文件，开始处理...")

    for filename in tqdm(files, desc="Processing"):
        file_path = os.path.join(input_dir, filename)

        try:
            # 1. 加载 NIfTI 数据
            img = nib.load(file_path)
            data = img.get_fdata()

            # 2. 确定输出子文件夹路径 (去除后缀作为文件夹名)
            # 例如: sample.nii.gz -> sample
            base_name = filename.replace('.nii.gz', '').replace('.nii', '')
            subject_output_dir = os.path.join(output_dir, base_name)
            os.makedirs(subject_output_dir, exist_ok=True)

            # 3. 遍历 Z 轴切片 (假设数据格式为 H x W x D)
            # 如果数据维度不同，可能需要调整 axis
            num_slices = data.shape[2] if len(data.shape) > 2 else 1

            for i in range(num_slices):
                slice_2d = data[:, :, i]

                # 可选：跳过全空切片以节省空间
                if np.max(slice_2d) == 0 and np.min(slice_2d) == 0:
                    continue

                # 4. 归一化并保存
                png_data = normalize_slice(slice_2d)
                save_path = os.path.join(subject_output_dir, f"slice_{i:03d}.png")
                imageio.imwrite(save_path, png_data)

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    print(f"处理完成！结果已保存至: {output_dir}")


# === 使用示例 ===
if __name__ == "__main__":
    # 修改为你的实际路径
    source_folder = "D:\Qin\dataset\MM-WHS 2017 Dataset\ct_train"  # 包含 .nii 文件的文件夹
    target_folder = "D:\\Qin\\project\\python\\unet\\mmwhsct\\train_data"  # 输出 PNG 的文件夹

    batch_nii_to_png(source_folder, target_folder)
