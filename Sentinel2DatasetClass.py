import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import albumentations as A
from albumentations import Compose, HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast, GaussianBlur
from albumentations.pytorch import ToTensorV2

# Reads preprocessed .npy images
# Loads images and labels based on index
class Sentinel2Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.augment = augment

        # Add augmentation
        self.augmentation = Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(p=0.2),
        ])

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

    # Define augmentations
        if self.augment:
            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.2),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Construct the full paths for the images and labels
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        # Load image and label
        image = np.load(image_path).astype(np.float32)  # Shape: [32, 32, 12]
        label = np.load(label_path).astype(np.float32)  # Shape: [32, 32]


        # Add normalization for each band
        for band in range(image.shape[2]):
            band_data = image[:, :, band]
            band_min, band_max = band_data.min(), band_data.max()
            if band_max > band_min:  
                image[:, :, band] = (band_data - band_min) / (band_max - band_min)
        
        # Add nodata handling
        mask = label != 255
        label[~mask] = 0
        
        if np.sum(mask) < 0.5 * label.size:
            return self.__getitem__((idx + 1) % len(self))
        
        # Apply augmentations if enabled
        if self.augment:
            # Reshape label to add channel dimension for albumentations
            label = np.expand_dims(label, axis=2)
            augmented = self.aug_transform(image=image, mask=label)
            image = augmented['image']
            label = augmented['mask'].squeeze()
        
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
