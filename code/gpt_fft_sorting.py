import os
import numpy as np
import cv2
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import shutil
from pathlib import Path
import pandas as pd

class FFTImageClusterer:
    def __init__(self, input_dir, output_base_dir, n_clusters=3):
        """
        Initialize the clusterer
        
        Parameters:
        input_dir (str): Directory containing input images
        output_base_dir (str): Base directory for sorted outputs
        n_clusters (int): Number of clusters to create
        """
        self.input_dir = input_dir
        self.output_base_dir = output_base_dir
        self.n_clusters = n_clusters

        
    def extract_fft_features(self, image_path):
        """
        Extract FFT-based features from an image
        """
        # Read image
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
            
        # Apply FFT
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log1p(np.abs(f_shift))
        
        # Normalize
        magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / \
                           (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))
        
        # Image center
        center_y, center_x = magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2
        y, x = np.ogrid[-center_y:magnitude_spectrum.shape[0]-center_y,
                       -center_x:magnitude_spectrum.shape[1]-center_x]
        radius = np.sqrt(x*x + y*y)
        
        # Extract features
        features = []
        
        # 1. Low Frequency Energy
        radius_threshold = min(center_x, center_y) * 0.1
        low_freq_mask = radius <= radius_threshold
        features.append(np.sum(magnitude_spectrum[low_freq_mask]) / np.sum(low_freq_mask))
        
        # 2. Radial Energy Distribution
        radial_profile = ndimage.mean(magnitude_spectrum, labels=radius.astype(int),
                                    index=np.arange(0, int(radius.max())+1))
        features.append(np.mean(np.diff(radial_profile)))
        
        # 3. Directional Energy (Anisotropy)
        vertical = np.sum(magnitude_spectrum[center_y-20:center_y+20, center_x-5:center_x+5])
        horizontal = np.sum(magnitude_spectrum[center_y-5:center_y+5, center_x-20:center_x+20])
        features.append(abs(vertical - horizontal) / (vertical + horizontal))
        
        # 4. Spectral Width
        peak_value = magnitude_spectrum[center_y, center_x]
        half_max_mask = magnitude_spectrum >= (peak_value / 2)
        features.append(np.sum(half_max_mask))
        
        # 5. Peak-to-Background Ratio
        background = np.mean(magnitude_spectrum[radius > radius_threshold])
        features.append(peak_value / background if background > 0 else 100)
        
        # 6. High Frequency Energy
        outer_radius = min(center_x, center_y) * 0.8
        high_freq_mask = radius >= outer_radius
        features.append(np.sum(magnitude_spectrum[high_freq_mask]) / np.sum(high_freq_mask))
        
        return np.array(features)
    
    def process_directory(self):
        """
        Process all images in the directory and cluster them
        """
        # Get all image files
        image_files = []
        features_list = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        
        for ext in valid_extensions:
            image_files.extend(Path(self.input_dir).glob(f'*{ext}'))
            image_files.extend(Path(self.input_dir).glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise ValueError("No image files found in the input directory")
            
        # Extract features for each image
        valid_files = []
        for img_path in image_files:
            features = self.extract_fft_features(img_path)
            if features is not None:
                features_list.append(features)
                valid_files.append(img_path)
        
        if not features_list:
            raise ValueError("No valid images could be processed")
            
        # Normalize features
        features_array = np.array(features_list)
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_array)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(normalized_features)
        
        # Create output directories and move files
        for cluster_idx in range(self.n_clusters):
            cluster_dir = Path(self.output_base_dir) / f'cluster_{cluster_idx}'
            cluster_dir.mkdir(parents=True, exist_ok=True)
            
            # Move files belonging to this cluster
            for file_path, cluster_label in zip(valid_files, clusters):
                if cluster_label == cluster_idx:
                    shutil.copy2(
                        str(file_path),
                        str(cluster_dir / file_path.name)
                    )

                    # adding the cluster info to the csv:
                    # clster_label - the cluster of the image
                    print(f'DEBUG: PATH/IMAGE NAME: {file_path.name}')
                    file_name = file_path.name.split()[0]
                    df = pd.read_csv('cloud_data_fft.csv')
                    df.loc[df['name'] == file_name, 'cluster'] = cluster_label
                    print(f'DEBUG: CLUSTER LABEL: {cluster_label}')
                    print(f'DEBUG: CLUSTER IDX: {cluster_idx}')

        
        return clusters, valid_files

def main():
    # Example usage
    input_directory = "target_images"
    output_directory = "cloude_sorted_images"
    n_clusters = 10  # Adjust based on your needs    
    
    clusterer = FFTImageClusterer(input_directory, output_directory, n_clusters)
    
    try:
        clusters, files = clusterer.process_directory()
        print(f"Successfully processed {len(files)} images into {n_clusters} clusters")
        for i in range(n_clusters):
            count = np.sum(clusters == i)
            print(f"Cluster {i}: {count} images")
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()