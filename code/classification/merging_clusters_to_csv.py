from pathlib import Path
import pandas as pd
from tqdm import tqdm

def find_image(image_name, search_dir):
    """
    Search for an image file by name in a directory (including subdirectories).
    
    Parameters:
    - image_name (str): The name of the image file (e.g., 'example.jpg').
    - search_dir (str or Path): The root directory to search in.
    
    Returns:
    - str: The full path of the image if found, else None.
    """
    search_path = Path(search_dir)
    
    for file in search_path.rglob(f'{image_name}.jpeg'):  # Recursively search for the file
        print(file.parent.name[-1])
        return file.parent.name[-1]  # Return the first match
    
    return None  # Return None if not found

# find_image('20190804-12-11-85-84.jpeg', 'pca_clusters')


df = pd.read_csv('cloud_data_fft.csv')

df['cluster'] = df['name'].apply(find_image, search_dir='pca_clusters')
df.to_csv('cloud_data_fft.csv', index=False)

print(df)
