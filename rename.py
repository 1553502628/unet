import os


def rename_files_remove_last_four_chars(folder_path):
    """
    重命名文件夹中的所有文件，去掉文件名最后的四个字符

    参数:
        folder_path (str): 文件夹路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 确保是文件而不是目录
        if os.path.isfile(os.path.join(folder_path, filename)):
            # 去掉文件名最后的四个字符
            new_name = filename[:-9] + '.png'

            # 确保新文件名与原文件名不同
            if new_name != filename:
                source_path = os.path.join(folder_path, filename)
                target_path = os.path.join(folder_path, new_name)

                try:
                    os.rename(source_path, target_path)
                    print(f"已重命名: {filename} -> {new_name}")
                except Exception as e:
                    print(f"重命名文件 {filename} 时出错: {e}")


# 使用示例
if __name__ == "__main__":
    folder = 'D:\Qin\dataset\Dataset_BUSI_with_GT\do_n\mask'

    rename_files_remove_last_four_chars(folder)
    print("操作完成！")