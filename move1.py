import os
import shutil
import random
from pathlib import Path


def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8):
    """
    划分数据集到train/val目录
    :param images_dir: 原始图像目录
    :param labels_dir: 原始标签目录
    :param output_dir: 输出根目录
    :param train_ratio: 训练集比例
    """
    # 创建目标目录结构
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dirs = [
        output_dir + '/images/train',
        output_dir + '/images/val',
        output_dir + '/labels/train',
        output_dir + '/labels/val'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    # 获取所有图像文件名（不带扩展名）
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    base_names = [os.path.splitext(f)[0] for f in image_files]

    # 过滤掉没有对应标签的图像
    valid_names = []
    for name in base_names:
        label_path = os.path.join(labels_dir, f"{name}.png")
        if os.path.exists(label_path):
            valid_names.append(name)
        else:
            print(f"警告: 缺少标签文件 {name}.png，跳过该图像")

    # 随机打乱数据集
    random.shuffle(valid_names)

    # 计算划分点
    split_idx = int(len(valid_names) * train_ratio)
    train_names = valid_names[:split_idx]
    val_names = valid_names[split_idx:]

    print(f"数据集统计:")
    print(f"总样本: {len(valid_names)}")
    print(f"训练集: {len(train_names)}")
    print(f"验证集: {len(val_names)}")

    # 复制训练集
    for name in train_names:
        # 复制图像
        for ext in ['.jpg', '.jpeg', '.png']:
            src_img = os.path.join(images_dir, f"{name}{ext}")
            if os.path.exists(src_img):
                shutil.copy(src_img, f"{output_dir}/images/train/{name}{ext}")
                break

        # 复制标签
        src_label = os.path.join(labels_dir, f"{name}.png")
        shutil.copy(src_label, f"{output_dir}/labels/train/{name}.png")

    # 复制验证集
    for name in val_names:
        # 复制图像
        for ext in ['.jpg', '.jpeg', '.png']:
            src_img = os.path.join(images_dir, f"{name}{ext}")
            if os.path.exists(src_img):
                shutil.copy(src_img, f"{output_dir}/images/val/{name}{ext}")
                break

        # 复制标签
        src_label = os.path.join(labels_dir, f"{name}.png")
        shutil.copy(src_label, f"{output_dir}/labels/val/{name}.png")

    print("数据集划分完成！")


# 使用示例
if __name__ == "__main__":
    # 配置路径（根据实际情况修改）
    IMAGES_DIR = "D:\Qin\dataset\Dataset_BUSI_with_GT\do_n\images"  # 原始图像目录
    LABELS_DIR = "D:\Qin\dataset\Dataset_BUSI_with_GT\do_n\mask"  # 原始标签目录
    OUTPUT_DIR = "D:\Qin\dataset\BUSIN"  # 输出目录

    split_dataset(IMAGES_DIR, LABELS_DIR, OUTPUT_DIR)
