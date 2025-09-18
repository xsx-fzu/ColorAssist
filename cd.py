
import cv2
import numpy as np
import torch

def rgb_to_lab(img):
    # img = img.cpu()
    # img = img.numpy()  # 将PyTorch张量转换为NumPy数组
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)  # 使用OpenCV进行颜色空间转换
    return torch.from_numpy(lab_img)  # 将NumPy数组转换回PyTorch张量

def calculate_cd(img1, img2):
    # 转换图像到Lab颜色空间
    lab1 = rgb_to_lab(img1) #(H,W,3)
    lab2 = rgb_to_lab(img2) #(H,W,3)
    # 计算色差
    chrom_diff = torch.sqrt(torch.sum((lab1[:, :, 1:] - lab2[:, :, 1:]) ** 2, dim=2))
    return torch.mean(chrom_diff)

def calculate_GCD(image):
    lab_image = rgb_to_lab(image)
    l_channel = lab_image[:, :, 0]  # L* channel
    a_channel = lab_image[:, :, 1]  # a* channel
    b_channel = lab_image[:, :, 2]  # b* channel

    height, width = l_channel.shape
    gcd_values = np.zeros((height, width))

    # Calculate GCD for each pixel
    for i in range(height):
        for j in range(width):
            li = l_channel[i, j]
            ai = a_channel[i, j]
            bi = b_channel[i, j]

            # Initialize sum of color differences
            gcd_sum = 0

            # Compare with all other pixels
            for m in range(height):
                for n in range(width):
                    lj = l_channel[m, n]
                    aj = a_channel[m, n]
                    bj = b_channel[m, n]

                    # Compute the Euclidean distance in LAB color space
                    gcd_sum += np.sqrt((li - lj)**2 + (ai - aj)**2 + (bi - bj)**2)

            # Normalize by the number of pixel pairs
            gcd_values[i, j] = gcd_sum / (height * width)

    return gcd_values