import numpy as np
import cv2
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import time
import pandas as pd


'''

This script is used to cluster the images using PCA and K-Means, the values of the clusters are then saved to the cloud_data_fft.csv file.
The images used for the PCA analysis are the fft magnitude spectrum images. (NOT THE ORIGINAL IMAGES)

'''
logger = logging.getLogger(__name__)
logging.basicConfig(filename='PCA.log', level=logging.INFO)


class ImagePCAClusterer:
    def __init__(self, input_dir, output_dir, n_components=100, n_clusters=10):
        """
        Initialize the PCA-based image clusterer.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.image_size = (444, 444)  # Fixed for your specific case

    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image.
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.info(f"Error processing {image_path}")
                return None

            if img.shape[:2] != self.image_size:
                img = cv2.resize(img, self.image_size)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.reshape(-1)  # Flatten image
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None

    def process_images_in_batches(self, batch_size):
        """
        Process images in batches, ensuring the last incomplete batch is dropped.
        """
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = sorted([
            file for ext in valid_extensions for file in Path(self.input_dir).glob(f'*{ext}')
        ])

        if not image_files:
            raise ValueError("No image files found!")

        # Drop last incomplete batch
        num_full_batches = len(image_files) // batch_size  # Number of full batches
        image_files = image_files[:num_full_batches * batch_size]  # Keep only full batches

        if batch_size < self.n_components:
            raise ValueError("Batch size must be at least as large as n_components.")

        ipca = IncrementalPCA(n_components=self.n_components)

        print("Processing images in batches...")
        valid_files = []
        transformed_data = []

        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i:i + batch_size]
            batch_data = []

            for img_path in batch_files:
                img_processed = self.load_and_preprocess_image(img_path)
                if img_processed is not None:
                    batch_data.append(img_processed)
                    valid_files.append(img_path)

            if batch_data:
                X_batch = np.array(batch_data)
                scaler = StandardScaler()
                X_batch_scaled = scaler.fit_transform(X_batch)

                ipca.partial_fit(X_batch_scaled)
                X_batch_pca = ipca.transform(X_batch_scaled)
                transformed_data.extend(X_batch_pca)

        if not transformed_data:
            raise ValueError("No valid images could be processed!")

        X_pca = np.array(transformed_data)

        print("Performing clustering...")
        kmeans = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=min(1000, len(X_pca)), random_state=42)
        clusters = kmeans.fit_predict(X_pca)

        print("Sorting images into clusters...")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        for cluster_idx in range(self.n_clusters):
            cluster_dir = Path(self.output_dir) / f'cluster_{cluster_idx}'
            cluster_dir.mkdir(exist_ok=True)

            for file_path, cluster_label in zip(valid_files, clusters):
                if cluster_label == cluster_idx:
                    shutil.copy2(str(file_path), str(cluster_dir / file_path.name))

        explained_variance = np.cumsum(ipca.explained_variance_ratio_)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance vs. Number of Components')
        plt.grid(True)
        plt.savefig(str(Path(self.output_dir) / 'explained_variance.png'))
        plt.close()

        return clusters, valid_files, ipca, kmeans


def main():
    config = {
        'input_dir': 'fft_magnitude_spectrum',
        'output_dir': 'pca_clusters',
        'n_clusters': 10
    }

    clusterer = ImagePCAClusterer(**config)

    try:
        clusters, files, ipca, kmeans = clusterer.process_images_in_batches(batch_size=100)

        print("\nClustering Summary:")
        print(f"Total images processed: {len(files)}")
        print(f"Explained variance with {clusterer.n_components} components: {np.sum(ipca.explained_variance_ratio_):.3f}")

        for i in range(clusterer.n_clusters):
            count = np.sum(clusters == i)
            print(f"Cluster {i}: {count} images")

    except Exception as e:
        print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    main()
