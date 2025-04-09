import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import random
import re
import time
import logging
import pandas as pd

'''

This script is used to take the original photos of the stratocumulus and analyze them using the Fast Fourier Transform (FFT) method.
The script will save the magnitude spectrum of the FFT as a figure and save the parameters extracted from the FFT analysis in a csv file.

'''

# ---------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(filename='fft.log', encoding='utf-8', level=logging.DEBUG)
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
    

    def fft_man(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("Image not found. Check the file path.")

        # Compute the 2D Fourier Transform and shift it to the center
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        # Compute the magnitude spectrum (log-scaled for visibility)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        # saving the magnitude spectrum as a figure
        if not os.path.exists('fft_magnitude_spectrum'):
            os.makedirs('fft_magnitude_spectrum')
        plt.figure(figsize=(10, 10))
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum')
        plt.axis('off')
        plt.savefig(f'fft_magnitude_spectrum/{self.name}_magnitude_spectrum.png')
        plt.close()

        # Get image size
        rows, cols = image.shape
        cx, cy = cols // 2, rows // 2  # Center coordinates

        ### 1. Extract DC Component (Zero Frequency)
        self.dc_component = np.abs(f_shift[cy, cx])

        ### 2. Extract Dominant Frequency (Highest Magnitude, excluding DC)
        f_shift_no_dc = f_shift.copy()
        f_shift_no_dc[cy, cx] = 0  # Remove DC component for analysis
        self.max_freq_indices = np.unravel_index(np.argmax(np.abs(f_shift_no_dc)), f_shift.shape)

        ### 3. Compute Low-Frequency and High-Frequency Energy
        low_freq_region = f_shift[cy-10:cy+10, cx-10:cx+10]  # 20x20 central region
        self.high_freq_energy = (
            np.sum(np.abs(f_shift[:10, :])) +  # Top rows
            np.sum(np.abs(f_shift[-10:, :])) +  # Bottom rows
            np.sum(np.abs(f_shift[:, :10])) +  # Left columns
            np.sum(np.abs(f_shift[:, -10:]))   # Right columns
        )

        self.low_freq_energy = np.sum(np.abs(low_freq_region))


        ### 4. Compute Radial Energy Distribution
        radius_values = np.linspace(1, min(cx, cy), num=50)  # 50 radial steps
        self.radial_energy = []

        for r in radius_values:
            mask = np.zeros_like(f_shift, dtype=np.uint8)
            y, x = np.ogrid[:rows, :cols]
            mask_area = (x - cx)**2 + (y - cy)**2 <= r**2
            mask[mask_area] = 1
            self.radial_energy.append(np.sum(np.abs(f_shift * mask)))

        ### 5. Extract Phase of the Dominant Frequency
        phase_spectrum = np.angle(f_shift)
        self.dominant_phase = phase_spectrum[self.max_freq_indices]

        ### Return all computed values
        return 


# Retrieving the names of the images from the directory (the name includes the time and location of the image)
def extract_jpeg_names(directory):
    jpeg_names = [f for f in os.listdir(directory) if f.endswith('.jpeg')]
    return jpeg_names # A list of the names of the images


def create_instances():
    instants = []
    directory_path = os.path.join('C:\\Users\\Temp\\Desktop', 'ran', 'code', 'target_images') # The location of the jpegs
    target_images = extract_jpeg_names(directory_path) # A list of the names of the images
    df = pd.DataFrame(columns=['name', 'year', 'month', 'day', 'minlat', 'maxlat', 'minlon', 'maxlon', 'dc_component', 'max_freq_indices', 'low_freq_energy', 'high_freq_energy', 'radial_energy', 'dominant_phase'])


    for file in target_images:
        instant = Instant(file)
        instant.fft_man(os.path.join(directory_path, file)) 
        instants.append(instant)
        
        # adding a print of the process to the log file
        logger.info(f'dc_component: {instant.dc_component}')
        logger.info(f'max_freq_indices: {instant.max_freq_indices}')
        logger.info(f'low_freq_energy: {instant.low_freq_energy}')
        logger.info(f'high_freq_energy: {instant.high_freq_energy}')
        logger.info(f'radial_energy: {instant.radial_energy}')
        logger.info(f'dominant_phase: {instant.dominant_phase}')
        # printing a progress bar
        total_files = len(target_images)
        processed_files = len(instants)
        progress = (processed_files / total_files) * 100
        print(f"Progress for fft: {progress:.2f}%")
        logger.info(f"Progress for fft: {progress:.2f}%")

        # for data conservation purposes, we will save the data in a csv file we created earlier df
        df = df._append({'name': instant.name, 'year': instant.year, 'month': instant.month, 'day': instant.day, 'minlat': instant.minlat, 'maxlat': instant.maxlat, 'minlon': instant.minlon, 'maxlon': instant.maxlon,
                         'dc_component': instant.dc_component,
                         'max_freq_indices': instant.max_freq_indices,
                         'low_freq_energy': instant.low_freq_energy,
                         'high_freq_energy': instant.high_freq_energy,
                         'radial_energy': instant.radial_energy,
                         'dominant_phase': instant.radial_energy
                         }, ignore_index=True)
        # saving the data in the csv file
    df.to_csv('cloud_data_fft.csv', index=False)

create_instances()




