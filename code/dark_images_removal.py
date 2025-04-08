import os
import cv2
import numpy as np
from shutil import copy2

# This code is used to sort images into two folders based on their brightness, for the rest of the code to be implemented only on the bright images (as the dark images are not useful for the analysis)
# The code is based on a threshold value, which is used to determine whether an image is dark or bright, and can be adjusted as needed.

def is_dark(image, threshold):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.sum(gray_image < 128)
    total_pixels = gray_image.size
    dark_ratio = dark_pixels / total_pixels
    return dark_ratio > threshold

def sort_images(input_folder, dark_folder, bright_folder, threshold=0.09):
    if not os.path.exists(dark_folder):
        os.makedirs(dark_folder)
    if not os.path.exists(bright_folder):
        os.makedirs(bright_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is not None:

                if is_dark(image, threshold):
                    copy2(image_path, dark_folder)
                else:
                    copy2(image_path, bright_folder)
            total_files = len([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
            processed_files = len(os.listdir(dark_folder)) + len(os.listdir(bright_folder))
            progress = (processed_files / total_files) * 100
            print(f"Progress: {progress:.2f}%")

if __name__ == "__main__":
    input_folder = os.path.join('C:\\Users\\Temp\Desktop', 'ran', 'closed_cells_ran')
    if not os.path.exists('dark_images'):
        os.makedirs('dark_images')
    dark_folder = "dark_images" # The dark image directory was manually deleted after the code was run
    if not os.path.exists('target_images'):
        os.makedirs('target_images')
    bright_folder = "target_images"
    sort_images(input_folder, dark_folder, bright_folder)