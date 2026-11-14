import os
from PIL import Image, ImageDraw, ImageFont


def add_text_to_images(
        input_folder,
        output_folder="processed_images",  # 默认输出文件夹名称
        text="Sample Text",
        position=(10, 10),
        font_path=None,
        font_size=10,
        text_color=(255, 0, 0),
        text_opacity=255
):
    """
    在文件夹中的所有图片的指定位置添加文字，并保存到新文件夹
    :param input_folder: 输入图片文件夹路径
    :param output_folder: 输出文件夹路径（默认 "processed_images"）
    :param text: 要添加的文字
    :param position: 文字位置 (x, y)，单位像素
    :param font_path: 字体文件路径（如 'arial.ttf'），若为 None 则使用默认字体
    :param font_size: 文字大小
    :param text_color: 文字颜色 (R, G, B)
    :param text_opacity: 文字透明度（0-255）
    """
    # 创建输出文件夹（如果不存在）
    os.makedirs(output_folder, exist_ok=True)

    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    # 遍历输入文件夹
    for img_name in os.listdir(input_folder):
        if img_name.lower().endswith(supported_formats):
            try:
                # 打开图片（确保是 RGBA 模式以支持透明度）
                img_path = os.path.join(input_folder, img_name)
                img = Image.open(img_path).convert("RGBA")

                # 创建文字图层
                txt_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(txt_layer)

                # 加载字体（失败时回退到默认字体）
                try:
                    font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
                except:
                    font = ImageFont.load_default()
                    print(f"警告：无法加载字体 '{font_path}'，已使用默认字体")

                # 设置文字颜色（含透明度）
                text_color_with_alpha = text_color + (text_opacity,)

                # 在指定位置添加文字
                draw.text(position, text, font=font, fill=text_color_with_alpha)

                # 合并原图和文字层
                result = Image.alpha_composite(img, txt_layer)

                # 保存到输出文件夹（转为 RGB 格式以兼容 JPG）
                output_path = os.path.join(output_folder, img_name)
                result.convert("RGB").save(output_path)
                print(f"已处理: {img_name} -> {output_path}")

            except Exception as e:
                print(f"处理 {img_name} 时出错: {str(e)}")


# 示例调用
if __name__ == "__main__":
    input_folder = "D:\Qin\对比结果\\aaa"  # 原始图片文件夹
    output_folder = "D:\Qin\对比结果\\test"  # 处理后的图片保存到此文件夹

    add_text_to_images(
        input_folder=input_folder,
        output_folder=output_folder,
        text="image",
        position=(310, 550),
        font_size=50,
        text_color=(255, 0, 0),
        text_opacity=255  # 半透明
    )