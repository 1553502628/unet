import os
import shutil


def move_mask_files(source_folder, target_folder):
    """
    将源文件夹中所有以'mask'结尾的文件移动到目标文件夹

    参数:
        source_folder (str): 源文件夹路径
        target_folder (str): 目标文件夹路径
    """
    # 确保目标文件夹存在，如果不存在则创建
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        # 检查文件名是否以'mask'结尾
        if filename.lower().endswith(').png'):
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)

            # 确保是文件而不是目录
            if os.path.isfile(source_path):
                try:
                    shutil.copy(source_path, target_path)
                    print(f"已移动: {filename}")
                except Exception as e:
                    print(f"移动文件 {filename} 时出错: {e}")


# 使用示例
if __name__ == "__main__":
    source = 'D:\Qin\dataset\Dataset_BUSI_with_GT\\normal'
    target = 'D:\Qin\dataset\Dataset_BUSI_with_GT\do_n\images'

    move_mask_files(source, target)
    print("操作完成！")