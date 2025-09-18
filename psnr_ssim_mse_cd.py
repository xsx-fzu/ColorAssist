import os
import numpy as np
from glob import glob
import cv2
from skimage.metrics import mean_squared_error, structural_similarity, peak_signal_noise_ratio


def read_img(path, color=True):
    if color:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def rgb_to_lab(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2Lab)


def chromatic_difference(img1, img2):
    lab1 = rgb_to_lab(img1)
    lab2 = rgb_to_lab(img2)
    chrom_diff = np.sqrt(np.sum((lab1[:, :, 1:] - lab2[:, :, 1:]) ** 2, axis=-1))
    return np.mean(chrom_diff)


def mse(tf_img1, tf_img2):
    return mean_squared_error(tf_img1, tf_img2)


def psnr(tf_img1, tf_img2):
    return peak_signal_noise_ratio(tf_img1, tf_img2)


def ssim(tf_img1, tf_img2):
    return structural_similarity(tf_img1, tf_img2, win_size=11, channel_axis=-1, data_range=1.0)


def main():
    WSI_MASK_PATH1 = 'XDU-CVDdataset/test/B'                           # 原图路径
    WSI_MASK_PATH2 = '10_XDU'            # 生成图路径

    # 支持多种图像格式
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

    path_real_all = []
    path_fake_all = []
    for ext in extensions:
        path_real_all += glob(os.path.join(WSI_MASK_PATH1, ext))
        path_fake_all += glob(os.path.join(WSI_MASK_PATH2, ext))

    # 创建“无扩展名”到路径的映射字典
    dict_real = {os.path.splitext(os.path.basename(p))[0]: p for p in path_real_all}
    dict_fake = {os.path.splitext(os.path.basename(p))[0]: p for p in path_fake_all}

    # 找出共同拥有的图像（按主名匹配）
    common_names = sorted(set(dict_real.keys()) & set(dict_fake.keys()))

    print("真实图像数量:", len(dict_real))
    print("生成图像数量:", len(dict_fake))
    print("匹配成功的图像数量:", len(common_names))

    if len(common_names) == 0:
        print("没有找到匹配的图像文件，确保图像主名一致即可，不必扩展名一致。")
        return

    list_psnr = []
    list_ssim = []
    list_mse = []
    list_chrom_diff = []

    for i, name in enumerate(common_names):
        path_real = dict_real[name]
        path_fake = dict_fake[name]

        t1 = read_img(path_real, color=True)
        t2 = read_img(path_fake, color=True)

        if t1 is None or t2 is None:
            print("图像读取失败:", path_real, path_fake)
            continue

        t1 = cv2.resize(t1, (128, 128))
        t2 = cv2.resize(t2, (128, 128))

        result1 = np.zeros(t1.shape, dtype=np.float32)
        result2 = np.zeros(t2.shape, dtype=np.float32)

        cv2.normalize(t1, result1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        cv2.normalize(t2, result2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        mse_num = mse(result1, result2)
        psnr_num = psnr(result1, result2)
        ssim_num = ssim(result1, result2)
        chrom_diff_num = chromatic_difference(t1, t2)

        list_psnr.append(psnr_num)
        list_ssim.append(ssim_num)
        list_mse.append(mse_num)
        list_chrom_diff.append(chrom_diff_num)

        print(f"{i + 1}/{len(common_names)}")
        print("图像:", name)
        print("PSNR:", psnr_num)
        print("SSIM:", ssim_num)
        print("MSE:", mse_num)
        print("Chromatic Difference:", chrom_diff_num)

    print("\n=== 平均指标 ===")
    print("平均 PSNR:", np.mean(list_psnr))
    print("平均 SSIM:", np.mean(list_ssim))
    print("平均 MSE:", np.mean(list_mse))
    print("平均 Chromatic Difference:", np.mean(list_chrom_diff))


if __name__ == '__main__':
    main()
