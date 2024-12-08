import os
import numpy as np
import torch
from torch.utils.data import Dataset

# reads preprocessed .npy images 
# loads images and labels based on index
class Sentinel2Dataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.npy')]
        self.label_files = [f.replace("image_", "label_") for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and label
        image = np.load(os.path.join(self.image_dir, self.image_files[idx]))
        label = np.load(os.path.join(self.label_dir, self.label_files[idx]))

        # Convert to PyTorch tensors
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label
