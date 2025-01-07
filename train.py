import os
import torch
from torch.utils.data import DataLoader
from Sentinel2DatasetClass import Sentinel2Dataset
from model import get_model, get_optimizer, get_loss_fn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau


# for visualisation
import matplotlib.pyplot as plt
import numpy as np
def train_model(train_dataset, val_dataset, batch_size, num_epochs, learning_rate, weight_decay, save_dir="models"):
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer setup
    model = get_model()
    optimizer = get_optimizer(model, learning_rate, weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    loss_fn = get_loss_fn()

    # Early stopping parameters
    patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Lists to track metrics
    train_losses = []
    val_losses = []

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training phase
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images).squeeze(1)
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Calculate average training loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss = validate_model(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break
            
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
            }, os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth"))

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'))
    plt.close()

    print("Training complete.")
    print(f"Best Validation Loss: {best_val_loss:.4f}")

def validate_model(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze(1)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
    
    return total_loss / num_batches




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
    train_dataset = Sentinel2Dataset(image_dir="data/processed/train", label_dir="data/processed/train", augment=True)
    val_dataset = Sentinel2Dataset(image_dir="data/processed/validation", label_dir="data/processed/validation", augment=False)

    # Training parameters
    initial_learning_rate = 0.0001
    weight_decay = 1e-4
    num_epochs = 15
    batch_size = 16

    # Train the model
    train_model(
        train_dataset,
        val_dataset,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=initial_learning_rate,
        weight_decay=weight_decay
    )

    # Visualization
    number_visu = 5
    visualize_images(val_dataset, number_visu)