import os
import torch
from torch.utils.data import DataLoader
from Sentinel2DatasetClass import Sentinel2Dataset
from model import get_model, get_optimizer, get_loss_fn
from tqdm import tqdm
# for visualisation
import matplotlib.pyplot as plt
import numpy as np
def train_model(train_dataset, val_dataset, batch_size=8, num_epochs=10, learning_rate=0.001, save_dir="models"):
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = get_model()
    optimizer = get_optimizer(model, learning_rate)
    loss_fn = get_loss_fn()

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(images).squeeze(1) # Resize for loss to function properly
            loss = loss_fn(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Validation
        if (epoch+1) % 2 == 0:
            validate_model(model, val_loader, loss_fn, device)

        # Save the model
        if (epoch+1) % 5 == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))

    print("Training complete.")

def validate_model(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).squeeze(1) # Resize for loss to function properly
            loss = loss_fn(outputs, labels)

            val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader)}")




# Function to visualize images, labels, and segmentation outputs
def visualize_images(val_dataset, num_images):
    """
    Visualize multiple images, their corresponding labels, and output segmentation maps.
    
    Parameters:
    - val_dataset
    - num_images: Number of images to visualize.
    """
    num_images = min(num_images, len(val_dataset))  # Ensure we don't exceed validation dataset length

    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))  # 3 columns: Image, Label, Output
    for i in range(num_images):
        # Choose random images in the dataset
        idx = torch.randint(0, len(val_dataset), (1,))
        print("index image to visualise=",idx)
        data, target = val_dataset.__getitem__(idx)
        # Extract RGB channels for the image
        rgb_data = data[ [3, 2, 1], ...]  # B4 (Red), B3 (Green), B2 (Blue)
        rgb_normalized = (rgb_data - rgb_data.min()) / (rgb_data.max() - rgb_data.min())  # Normalize

        # Plot Input Image (RGB)
        axes[i, 0].imshow(rgb_normalized.permute(1, 2, 0).numpy())  # Convert CHW -> HWC
        axes[i, 0].set_title(f"Input Image {idx} (RGB)")
        axes[i, 0].axis("off")

        # Plot Label (Ground Truth)
        axes[i, 1].imshow(target.numpy(), cmap="viridis")
        axes[i, 1].set_title(f"Label {idx}")
        axes[i, 1].axis("off")

        # Plot Output Segmentation Map
        model=get_model()
        output_image=model(data)
        output_2d = output_image.squeeze().detach().numpy()  # Remove channel dimension
        axes[i, 2].imshow(output_2d, cmap="viridis")
        axes[i, 2].set_title(f"Output {idx}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load datasets
    train_dataset = Sentinel2Dataset(image_dir="data/processed/train", label_dir="data/processed/train")
    val_dataset = Sentinel2Dataset(image_dir="data/processed/validation", label_dir="data/processed/validation")

    # Train the model
    train_model(train_dataset, val_dataset, batch_size=8, num_epochs=10, learning_rate=0.001)

    number_visu=5
    
    visualize_images(val_dataset,number_visu)

