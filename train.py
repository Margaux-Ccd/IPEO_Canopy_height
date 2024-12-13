import os
import torch
from torch.utils.data import DataLoader
from Sentinel2DatasetClass import Sentinel2Dataset
from model import get_model, get_optimizer, get_loss_fn
from tqdm import tqdm

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

if __name__ == "__main__":
    # Load datasets
    train_dataset = Sentinel2Dataset(image_dir="data/processed/train", label_dir="data/processed/train")
    val_dataset = Sentinel2Dataset(image_dir="data/processed/validation", label_dir="data/processed/validation")

    # Train the model
    train_model(train_dataset, val_dataset, batch_size=8, num_epochs=10, learning_rate=0.001)
