import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast, GaussianBlur
from albumentations.pytorch import ToTensorV2

# Reads preprocessed .npy images
# Loads images and labels based on index
class Sentinel2Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # List only image files (image_*.npy)
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.npy') and f.startswith('image_')]
        
        # List corresponding label files (label_*.npy)
        self.label_files = [f for f in os.listdir(label_dir) if f.endswith('.npy') and f.startswith('label_')]

        # Ensure the number of images and labels match
        assert len(self.image_files) == len(self.label_files), \
            f"Number of images ({len(self.image_files)}) and labels ({len(self.label_files)}) doesn't match."

        # Sort files to ensure alignment
        self.image_files.sort()
        self.label_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Construct the full paths for the images and labels
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # Load image and label
        image = np.load(image_path).astype(np.float32)  # Shape: [32, 32, 12]
        label = np.load(label_path).astype(np.float32)  # Shape: [32, 32]

        # Ensure image has the correct shape (32, 32, 12)
        if len(image.shape) == 3 and image.shape[2] == 12:
            image = np.transpose(image, (2, 0, 1))  # Shape: [12, 32, 32]
        else:
            print(f"Unexpected image shape: {image.shape}. Expected (32, 32, 12)")

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
