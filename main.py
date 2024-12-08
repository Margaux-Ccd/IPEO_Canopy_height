import os
from preprocessing import split_and_preprocess_data
from Sentinel2DatasetClass import Sentinel2Dataset

# Runs preprocessing and loads images and corresponding labels 
if __name__ == "__main__":
    # Directories
    data_dir = "data/canopy_height_dataset"
    output_dir = "data/processed"
    csv_path = os.path.join(data_dir, "data_split.csv")

    # Preprocess the data
    split_and_preprocess_data(csv_path, data_dir, output_dir, sample_size=50)
    
    # use Sentinel2Dataset to load the processed data
    train_dataset = Sentinel2Dataset(
        image_dir=os.path.join(output_dir, 'train'),
        label_dir=os.path.join(output_dir, 'train'),
    )
    
    # Print first item in the dataset as a test
    image, label = train_dataset[0]
    print(f"Image shape: {image.shape}, Label shape: {label.shape}")