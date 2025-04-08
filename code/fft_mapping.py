import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import os
from scipy import ndimage
import networkx as nx


def apply_lowpass_filter(image, sigma=1.0):
    
    filtered_image = ndimage.gaussian_filter(image, sigma=sigma)
        
    return filtered_image


def generate_fft_frequency_heatmap(image_path, block_size=44, log_scale=True, sigma=1.0):
    """
    Generate a heatmap showing frequency distribution of image blocks using FFT.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file
    block_size : int
        Size of blocks to analyze (e.g., 44 means 44x44 pixel blocks)
    log_scale : bool
        Whether to use log scale for the FFT magnitude (recommended)
    
    Returns:
    --------
    numpy.ndarray
        The frequency heatmap array
    """
    # Load the image
    img = Image.open(image_path)
    
    # Convert to grayscale if it's not already
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to numpy array
    img_array_original = np.array(img).astype(float)

    # applying a lowpass filter
    img_array = apply_lowpass_filter(img_array_original, sigma=sigma)

    # exporting the original image with and without the lowpass filter applied
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_array_original, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(img_array, cmap='gray')
    axes[1].set_title('Lowpass Filter Applied')
    plt.tight_layout()
    plt.show()

    
    # Get image dimensions
    height, width = img_array.shape
    
    # Calculate number of blocks in each dimension
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size
    
    # Initialize heatmap array
    heatmap = np.zeros((num_blocks_y, num_blocks_x))
    
    # Process each block with FFT
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            # Extract the block
            block = img_array[y*block_size:(y+1)*block_size, 
                             x*block_size:(x+1)*block_size]
            
            # Apply window function to reduce edge effects
            window = np.hanning(block_size)[:, np.newaxis] * np.hanning(block_size)
            windowed_block = block * window
            
            # Apply 2D FFT
            fft_block = np.fft.fft2(windowed_block)
            
            # Shift zero frequency to center
            fft_shifted = np.fft.fftshift(fft_block)
            
            # Calculate magnitude spectrum
            magnitude_spectrum = np.abs(fft_shifted)
            
            # Apply log scale if requested
            if log_scale:
                # Add small constant to avoid log(0)
                magnitude_spectrum = np.log1p(magnitude_spectrum)
            
            # Exclude the DC component (very low frequencies at the center)
            center = block_size // 2
            mask_size = 3  # Size of the center mask
            mask = np.ones_like(magnitude_spectrum, dtype=bool)
            mask[center-mask_size:center+mask_size+1, center-mask_size:center+mask_size+1] = False
            
            # Calculate the mean high-frequency component as our measure
            heatmap[y, x] = np.mean(magnitude_spectrum[mask])
    
    # Plot the heatmap
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(heatmap, cmap='viridis', cbar_kws={'label': 'Average Frequency Magnitude'})
    plt.title(f'FFT Frequency Heatmap (Block Size: {block_size}x{block_size})')
    plt.xlabel('X Blocks')
    plt.ylabel('Y Blocks')
    plt.savefig('cloud_frequency_heatmap/fft_frequency_heatmap.png')
    plt.show()
    
    return heatmap

def visualize_frequency_bands(image_path, block_size=44, num_bands=4, sigma=1.0):
    """
    Analyze and visualize different frequency bands in the image.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file
    block_size : int
        Size of blocks to analyze
    num_bands : int
        Number of frequency bands to divide the spectrum into
    """
    # Load the image
    img = Image.open(image_path)
    
    # Convert to grayscale if it's not already
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to numpy array
    img_array = np.array(img).astype(float)

    # applying the lowpass filter
    img_array = apply_lowpass_filter(img_array, sigma=sigma)
    
    # Get image dimensions
    height, width = img_array.shape
    
    # Calculate number of blocks in each dimension
    num_blocks_y = height // block_size
    num_blocks_x = width // block_size
    
    # Initialize heatmaps for different frequency bands
    heatmaps = [np.zeros((num_blocks_y, num_blocks_x)) for _ in range(num_bands)]
    
    # Process each block
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            # Extract the block
            block = img_array[y*block_size:(y+1)*block_size, 
                             x*block_size:(x+1)*block_size]
            
            # Apply window function
            window = np.hanning(block_size)[:, np.newaxis] * np.hanning(block_size)
            windowed_block = block * window
            
            # Apply 2D FFT
            fft_block = np.fft.fft2(windowed_block)
            fft_shifted = np.fft.fftshift(fft_block)
            magnitude_spectrum = np.log1p(np.abs(fft_shifted))
            
            # Create frequency bands
            # Generate masks for different frequency bands (distance from center)
            center = block_size // 2
            max_radius = np.sqrt(2) * center  # Maximum distance from center
            
            for band in range(num_bands):
                inner_radius = (band / num_bands) * max_radius
                outer_radius = ((band + 1) / num_bands) * max_radius
                
                # Create distance matrix from center
                y_indices, x_indices = np.ogrid[:block_size, :block_size]
                distance_from_center = np.sqrt((y_indices - center)**2 + (x_indices - center)**2)
                
                # Create band mask
                band_mask = (distance_from_center >= inner_radius) & (distance_from_center < outer_radius)
                
                # Calculate average magnitude for this band
                heatmaps[band][y, x] = np.mean(magnitude_spectrum[band_mask])
    
    # Plot all heatmaps in a grid
    fig, axes = plt.subplots(1, num_bands, figsize=(5*num_bands, 5))
    
    if num_bands == 1:
        axes = [axes]
    
    for band, (heatmap, ax) in enumerate(zip(heatmaps, axes)):
        sns.heatmap(heatmap, ax=ax, cmap='viridis')
        if band == 0:
            band_name = "Low"
        elif band == num_bands - 1:
            band_name = "High"
        else:
            band_name = f"Mid-{band}"
        ax.set_title(f'{band_name} Frequency Band')
        ax.set_xlabel('X Blocks')
        ax.set_ylabel('Y Blocks')
    
    plt.tight_layout()
    plt.savefig('cloud_frequency_heatmap/frequency_bands.png')
    plt.show()
    
    return heatmaps



# Example usage
if __name__ == "__main__":

    # creating a directory to store the outputs:
    if not os.path.exists('cloud_frequency_heatmap'):
        os.makedirs('cloud_frequency_heatmap')

    # Replace with your image path
    image_path = "black_images_removed/20190801-12-11-93-92.jpeg"
    
    # Generate FFT heatmap with 44x44 pixel blocks
    # This value divides 444 evenly, resulting in a 10x10 heatmap
    heatmap = generate_fft_frequency_heatmap(image_path, block_size=10, sigma=3.0)
    
    # Visualize different frequency bands
    visualize_frequency_bands(image_path, block_size=10, num_bands=4, sigma=3.0)
