import os
import numpy as np
import pandas as pd
from PIL import Image
import rasterio

def fix_csv_paths(csv_path, root_dir):
    # Load the CSV
    df = pd.read_csv(csv_path) 
    
    
    # Fix label paths (replace 'images' with 'labels')
    df['label_path'] = df['label_path'].str.replace('images/', 'labels/')

    # Prepend root directory to paths
    df['image_path'] = df['image_path'].apply(lambda x: os.path.join(root_dir, x))
    df['label_path'] = df['label_path'].apply(lambda x: os.path.join(root_dir, x))

    return df


def load_image(image_path):
    # Use rasterio to read multi-band images
    with rasterio.open(image_path) as src:
        image = src.read()  # Read all bands (returns shape: (bands, height, width))
    return image.transpose(1, 2, 0)  # Convert to (height, width, bands)


def load_label(label_path):
    # Read the label
    with rasterio.open(label_path) as src:
        label = src.read(1)  # Only read the first band for labels
    return label


# Get a subset of 50 images to practice with
def preprocess_subset(df, subset, output_dir, sample_size=50):
    # Filter dataset by subset (train/val/test)
    subset_df = df[df['split'] == subset].sample(n=sample_size, random_state=100)
    
    os.makedirs(output_dir, exist_ok=True)

    # Process each image-label pair
    for idx, row in subset_df.iterrows():
        # Load image and label
        image = load_image(row['image_path'])
        label = load_label(row['label_path'])

        # Save images and labels as npy files
        np.save(os.path.join(output_dir, f"image_{idx}.npy"), image)
        np.save(os.path.join(output_dir, f"label_{idx}.npy"), label)


def split_and_preprocess_data(csv_path, root_dir, output_dir, sample_size=50):
    # Fix paths in the CSV file
    df = fix_csv_paths(csv_path, root_dir)
    
    # Process train, validation, and test subsets
    preprocess_subset(df, 'train', output_dir=os.path.join(output_dir, 'train'), sample_size=sample_size)
    preprocess_subset(df, 'validation', output_dir=os.path.join(output_dir, 'validation'), sample_size=sample_size)
    preprocess_subset(df, 'test', output_dir=os.path.join(output_dir, 'test'), sample_size=sample_size)

