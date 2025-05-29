import os
import cv2
import numpy as np

# 定义掩码文件夹路径
mask_folder_path = "/mnt/data2/home/gaogeng/MiDSS-master/data/Polyp/Domain1/train/mask"  # 替换为你的文件夹路径

# 遍历文件夹中的所有PNG文件
for filename in os.listdir(mask_folder_path):
    if filename.endswith(".png"):  # 确保是PNG文件
        mask_path = os.path.join(mask_folder_path, filename)

        # 读取掩码图片
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 转换掩码：大于128设置为255，小于等于128设置为0
        mask = np.where(mask > 128, 255, 0).astype(np.uint8)

        # 将处理后的掩码替换原文件
        cv2.imwrite(mask_path, mask)

        print(f"已处理文件: {filename}")
