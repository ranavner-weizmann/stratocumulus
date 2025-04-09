import os
import cv2
import shutil
from pathlib import Path
from tqdm import tqdm

def has_black_pixels(image_path):
    """Check if an image contains any black pixels (0,0,0)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)  # Load as color image
    if img is None:
        return True  # Skip unreadable images
    
    return (img == [0, 0, 0]).all(axis=2).any()  # Check if any pixel is black

def copy_images_without_black_pixels(input_folder, output_folder):
    """Copy images from input_folder to output_folder if they contain no black pixels."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = list(input_path.glob("*.*"))  # Get all files
    
    copied_count = 0
    for image_file in tqdm(image_files, desc="Processing images"):
        if image_file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
            if not has_black_pixels(image_file):
                shutil.copy2(image_file, output_path / image_file.name)
                copied_count += 1
    
    print(f"Copied {copied_count} images to {output_folder}")

if __name__ == "__main__":
    # making sure the output folder exists and creating it if not:
    if not os.path.exists("../../data/black_images_removed"):
        os.makedirs("../../data/black_images_removed")
    input_folder = "../../data/target_images"  # Change this to your input folder
    output_folder = "../../data/black_images_removed"  # Change this to your output folder
    copy_images_without_black_pixels(input_folder, output_folder)
    print(os.listdir('../data'))
    print(os.getcwd())