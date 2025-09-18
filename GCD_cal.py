import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from PIL import Image
from skimage.color import rgb2lab

def calculate_GCD(image):
    lab_image = rgb2lab(image)
    l_channel = lab_image[:, :, 0]
    a_channel = lab_image[:, :, 1]
    b_channel = lab_image[:, :, 2]

    height, width = l_channel.shape
    gcd_values = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            li = l_channel[i, j]
            ai = a_channel[i, j]
            bi = b_channel[i, j]

            diff_l = li - l_channel
            diff_a = ai - a_channel
            diff_b = bi - b_channel

            distances = np.sqrt(diff_l**2 + diff_a**2 + diff_b**2)
            gcd_values[i, j] = np.mean(distances)

    return gcd_values

def calculate_GCD_folder2(folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    if not image_files:
        raise FileNotFoundError("No PNG images found in the folder.")

    average_GCD_values_all_images = []
    images_with_low_GCD = []

    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)

        
        image = Image.open(image_path).convert('RGB').resize((128, 128), Image.LANCZOS)
        image_np = np.array(image)

        gcd_values = calculate_GCD(image_np)
        average_GCD_value = np.mean(gcd_values)
        average_GCD_values_all_images.append(average_GCD_value)

        print(f"Average GCD value for image {idx + 1}: {average_GCD_value:.4f}")

        if average_GCD_value < 30:
            images_with_low_GCD.append(filename)

    overall_average_GCD = np.mean(average_GCD_values_all_images)
    print(f"Overall average GCD value for all images: {overall_average_GCD:.4f}")

    if images_with_low_GCD:
        print("Images with average GCD value less than 30:")
        for img_name in images_with_low_GCD:
            print(f" - {img_name}")
    else:
        print("No images with average GCD value less than 30.")
        
def calculate_GCD_folder_genrecolor(folder_path):
   
    image_files = [f for f in os.listdir(folder_path)
                   if 'recolorsim' in f and f.lower().endswith('.png')]

    if not image_files:
        raise FileNotFoundError("No '_recolorsim.png' images found in the folder.")

    average_GCD_values_all_images = []
    images_with_low_GCD = []

    for idx, filename in enumerate(image_files):
        image_path = os.path.join(folder_path, filename)

        try:
            image = Image.open(image_path).convert('RGB').resize((128, 128), Image.LANCZOS)
        except Exception as e:
            print(f"Failed to open {filename}: {e}")
            continue

        image_np = np.array(image)

        gcd_values = calculate_GCD(image_np)
        average_GCD_value = np.mean(gcd_values)
        average_GCD_values_all_images.append(average_GCD_value)

        print(f"Average GCD value for image {idx + 1} ({filename}): {average_GCD_value:.4f}")

        if average_GCD_value < 30:
            images_with_low_GCD.append(filename)

    overall_average_GCD = np.mean(average_GCD_values_all_images)
    print(f"\nOverall average GCD value for all '_genrecolor.png' images: {overall_average_GCD:.4f}")

    if images_with_low_GCD:
        print("Images with average GCD value less than 30:")
        for img_name in images_with_low_GCD:
            print(f" - {img_name}")
    else:
        print("No images with average GCD value less than 30.")


calculate_GCD_folder_genrecolor("/root/7.3T/XSX/ColorCorrect-ColorAssist/experiments_train_CVD_FZU_unet5CVDnet3_CVDP1_loss_simple0.5loss_contrast_global0.5loss/cvdcolor_250703_224431/results/1372/")
