import os
from PIL import Image


def rgb_to_hsl(r, g, b):
    # 归一化RGB值到[0, 1]范围
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # 计算最大值、最小值和差值
    c_max = max(r, g, b)
    c_min = min(r, g, b)
    delta = c_max - c_min

    # 计算色调H
    if delta == 0:
        h = 0
    elif c_max == r:
        h = (60 * ((g - b) / delta)) % 360
    elif c_max == g:
        h = (60 * ((b - r) / delta)) + 120
    else:
        h = (60 * ((r - g) / delta)) + 240

    # 计算亮度L
    l = (c_max + c_min) / 2

    # 计算饱和度S
    if delta == 0:
        s = 0
    else:
        s = delta / (1 - abs(2 * l - 1))

    # 将H、S、L值转换为[0, 1]范围
    return h, s, l


def convert_rgb_to_hsl(image_path):
    img = Image.open(image_path)

    # 确保是RGB图像
    if img.mode != 'RGB':
        raise ValueError(f"图像 {image_path} 不是RGB模式")

    # 获取图像的像素数据
    pixels = img.load()

    # 获取图像尺寸
    width, height = img.size

    # 创建一个新的HSL图像
    hsl_img = Image.new("RGB", img.size)

    # 遍历每个像素并转换RGB到HSL
    for i in range(width):
        for j in range(height):
            r, g, b = pixels[i, j]
            h, s, l = rgb_to_hsl(r, g, b)
            # 将H, S, L值映射到[0, 255]范围
            h = int(h / 360 * 255)
            s = int(s * 255)
            l = int(l * 255)
            hsl_img.putpixel((i, j), (h, s, l))

    return hsl_img


def process_images_in_folder(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 只处理图像文件（可以根据需要修改扩展名筛选）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                print(f"正在处理: {filename}")
                hsl_img = convert_rgb_to_hsl(file_path)

                # 保存HSL图像，替换原图像
                hsl_img.save(file_path)
                print(f"已替换图像: {filename}")
            except Exception as e:
                print(f"处理图像 {filename} 时出错: {e}")


if __name__ == "__main__":
    folder_path = '/mnt/data2/home/gaogeng/MiDSS-master/data/Fundus/Domain1/train/image'
    process_images_in_folder(folder_path)
