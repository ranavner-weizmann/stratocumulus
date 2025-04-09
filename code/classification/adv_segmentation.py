import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import random
import re
import time
import logging
import pandas as pd

# ---------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(filename='log_segmentation.log', encoding='utf-8', level=logging.DEBUG)
logger.info('SCRIPT STARTED, TIME: ' + str(time.time()))
# ---------------------------------

# The Instant class is used to contain the date and location of each jpeg file, plus the number of clouds and their average size
class Instant:

    NAME_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})-(\d+)-(\d+)-(\d+)-(\d+)")
    

    def __init__(self, file):
        self.extract_date_time(file)

    def extract_date_time(self, file):
        self.match = self.NAME_PATTERN.match(file)
        self.year = self.match.group(1)
        self.month = self.match.group(2)
        self.day = self.match.group(3)

        # inserting a condition that puts a minus sign before the minlat and minlon if they are negative
        # in august untill october the minlat and maxlat is negative, so we need to add a minus sign before it
        if self.month == '08' or self.month == '09' or self.month == '10':
            self.minlat = -int(self.match.group(4))
            self.maxlat = -int(self.match.group(5))
        else:
            self.minlat = self.match.group(4)
            self.maxlat = self.match.group(5)
        self.minlon = -int(self.match.group(6))
        self.maxlon = -int(self.match.group(7))
        # the name just until the first dot
        self.name = file.split('.')[0]
        # print(f'Year: {self.year}, Month: {self.month}, Day: {self.day}, Minlat: {self.minlat}, Maxlat: {self.maxlat}, Minlon: {self.minlon}, Maxlon: {self.maxlon}')
    

    def segment_and_count_clouds(self, image_path, thresh, maxdi, iter):
        # Load the image
        image = cv2.imread(image_path)
        original_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Increase contrast to improve thresholding
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold the image to create a binary image
        _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY) # ----------------------------------------------------

        # Perform morphological operations to remove small noise and separate clouds
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Distance transform to prepare for watershed
        dist_transform = cv2.distanceTransform(opened, cv2.DIST_L2, 5)

        # Threshold the distance transform to find sure foreground
        _, sure_fg = cv2.threshold(dist_transform, maxdi * dist_transform.max(), 255, 0)

        # Find sure background by dilating the binary image
        sure_bg = cv2.dilate(opened, kernel, iterations=iter)

        # Subtract sure foreground from sure background to get unknown regions
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Label the sure foreground regions
        _, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that the background is not zero
        markers = markers + 1

        # Mark the unknown regions with zero
        markers[unknown == 255] = 0

        # Apply the watershed algorithm
        markers = cv2.watershed(image, markers)

        # Create a copy of the original image for visualization
        segmented_image = original_image.copy()

        # Assign random colors to each segment
        unique_markers = np.unique(markers)
        for marker in unique_markers:
            if marker == -1:  # Skip boundaries
                continue
            mask = markers == marker
            color = np.random.randint(0, 255, size=(3,)).tolist()
            segmented_image[mask] = color

        # Annotate each segment with its number
        for marker in unique_markers:
            if marker <= 1:  # Skip background and boundaries
                continue
            mask = markers == marker
            coords = np.column_stack(np.where(mask))
            y, x = coords[coords.shape[0] // 2]  # Approximate center
            cv2.putText(segmented_image, str(marker - 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Count the number of distinct clouds (ignoring the background)
        num_clouds = len(unique_markers) - 2  # Subtract 2 for background and boundaries
        self.clouds_num = num_clouds

        # Let's add the average size of the clouds based on the number of pixels and the scale of the image
        scale = 0.0625  # The resolution is 250m, therefore each pixel area is 0.0625 km^2
        cloud_sizes = [np.sum(markers == marker) for marker in unique_markers if marker > 1]
        self.avg_size = np.median(cloud_sizes) * scale
        # print(f"Number of clouds: {num_clouds}, Average size: {self.avg_size:.2f} km^2")

        # Display the results
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        plt.subplot(1, 3, 2)
        plt.title("Binary Image")
        plt.imshow(binary, cmap='gray')

        # add the number of clouds to the image
        plt.text(0, 580, f"Number of clouds: {num_clouds}", color='white', backgroundcolor='black', fontsize=11)
        # add the parameters to the image
        plt.text(0, 630, f"Threshold: {thresh}, Max Distance: {maxdi}, Iterations: {iter}", color='white', backgroundcolor='black', fontsize=11)
        # add the average size of the clouds to the image
        plt.text(0, 680, f"Average size: {self.avg_size:.2f} km^2", color='white', backgroundcolor='black', fontsize=11)

        plt.subplot(1, 3, 3)
        plt.title("Segmented Image")
        plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))

        if not os.path.exists('segmented_images_visualization'):
            os.makedirs('segmented_images_visualization')
        plt.savefig(f'segmented_images_visualization\{self.year, self.month, self.day, self.minlat, self.maxlat, self.minlon, self.maxlon}.png')
    
        plt.close()
        return num_clouds



# Retrieving the names of the images from the directory (the name includes the time and location of the image)
def extract_jpeg_names(directory):
    jpeg_names = [f for f in os.listdir(directory) if f.endswith('.jpeg')]
    return jpeg_names # A list of the names of the images


def create_instances():
    instants = []
    directory_path = os.path.join('C:\\Users\\Temp\Desktop', 'ran', 'code', 'target_images') # The location of the jpegs
    target_images = extract_jpeg_names(directory_path) # A list of the names of the images
    df = pd.DataFrame(columns=['name', 'year', 'month', 'day', 'minlat', 'maxlat', 'minlon', 'maxlon', 'clouds_num', 'avg_size'])


    for file in target_images:
        instant = Instant(file)
        # instant.segment_and_count_clouds(os.path.join(directory_path, file), 135, 0.32, 1) # The parameters are the threshold, max distance, and iterations
        instant.segment_and_count_clouds(os.path.join(directory_path, file), 160, 0.5, 2) # The parameters are the threshold, max distance, and iterations
        instants.append(instant)
        
        # adding a print of the process to the log file
        logger.info(f'number of clouds in the image: {instant.clouds_num}')
        logger.info(f'average size of the clouds in the image: {instant.avg_size}')
        # printing a progress bar
        total_files = len(target_images)
        processed_files = len(instants)
        progress = (processed_files / total_files) * 100
        print(f"Progress for segmentation: {progress:.2f}%")
        logger.info(f"Progress for segmentation: {progress:.2f}%")

        # for data conservation purposes, we will save the data in a csv file we created earlier df
        df = df._append({'name': instant.name, 'year': instant.year, 'month': instant.month, 'day': instant.day, 'minlat': instant.minlat, 'maxlat': instant.maxlat, 'minlon': instant.minlon, 'maxlon': instant.maxlon, 'clouds_num': instant.clouds_num, 'avg_size': instant.avg_size}, ignore_index=True)
        # saving the data in the csv file
    df.to_csv('cloud_data.csv', index=False)

create_instances()




