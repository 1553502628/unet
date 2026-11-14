import os
from PIL import Image, ImageOps


def vertical_concatenate_folders_with_bottom_padding(
        folder1, folder2, output_folder,
        target_size=(340, 340), bottom_padding=180
):
    """
    将 folder1 的图片（保持原尺寸）与 folder2 的图片（调整为 340x340）垂直拼接，
    并在底部增加 180 像素的白色背景。
    :param folder1: 第一个文件夹路径（原尺寸图片）
    :param folder2: 第二个文件夹路径（调整为 340x340）
    :param output_folder: 输出文件夹路径
    :param target_size: 第二个文件夹图片的目标尺寸 (宽, 高)
    :param bottom_padding: 底部白色背景的高度（默认 180）
    """
    os.makedirs(output_folder, exist_ok=True)

    images1 = [f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images2 = [f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images1 = sorted(images1, key=lambda x: (len(x), x), reverse=False)
    print(images1)

    if len(images1) != len(images2):
        raise ValueError(f"图片数量不一致！folder1 有 {len(images1)} 张，folder2 有 {len(images2)} 张")

    for img1_name, img2_name in zip(images1, images2):
        img1_path = os.path.join(folder1, img1_name)
        img2_path = os.path.join(folder2, img2_name)

        try:
            # 打开图片（img1 保持原尺寸，img2 调整为 340x340）
            img1 = Image.open(img1_path)
            img2 = Image.open(img2_path).resize(target_size)

            # 计算最终画布的宽度（取 max(img1.width, 340)）
            final_width = max(img1.width, target_size[0])
            final_height = img1.height + target_size[1] + bottom_padding  # 增加底部补白

            # 创建新画布（白色背景）
            result = Image.new('RGB', (final_width, final_height), (255, 255, 255))

            # 粘贴 img1（保持原尺寸，水平居中）
            offset_x = (final_width - img1.width) // 2
            result.paste(img1, (offset_x, 0))

            # 粘贴 img2（340x340，水平居中）
            offset_x = (final_width - target_size[0]) // 2
            result.paste(img2, (offset_x, img1.height))

            # 保存结果（底部已自动填充白色）
            output_path = os.path.join(output_folder, img1_name)
            result.save(output_path)
            print(f"已保存: {output_path}")

        except Exception as e:
            print(f"处理 {img1_name} 和 {img2_name} 时出错: {str(e)}")


# 使用示例
folder1 = "D:\Qin\对比结果\新建文件夹\outputs"  # 替换为第一个文件夹路径
folder2 = "D:\Qin\project\python\\unet\prediction\PROMISE12\\unet22"  # 替换为第二个文件夹路径
output_folder = "D:\Qin\对比结果\\all"  # 替换为输出文件夹路径

vertical_concatenate_folders_with_bottom_padding(
    folder1, folder2, output_folder,
    target_size=(340, 340), bottom_padding=180
)