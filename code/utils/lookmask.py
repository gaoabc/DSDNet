import cv2
import numpy as np

# 读取掩码图片
mask_path = "/mnt/data2/home/gaogeng/MiDSS-master/data/EGC/Domain1/train/mask/patient1017.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# 获取掩码中的所有唯一值
unique_values, counts = np.unique(mask, return_counts=True)

# 输出所有不相同的值及其对应的计数
for value, count in zip(unique_values, counts):
    print(f"值 {value} 出现了 {count} 次")
