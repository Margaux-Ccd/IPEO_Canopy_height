import os
import zipfile
import gdown

def download_and_extract_data():
    # Project data url in gdrive
    url = "https://drive.google.com/uc?id=1iRQDJ4qCmUGrLyjgzeFcvxir90IlYohr"

    # Download the zip file
    output_path = "dataset.zip"
    gdown.download(url, output_path, quiet=False)

    # Extract the zip file
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall("data")
    print("Data extracted to 'data/' directory")

    zip_path = "data/canopy_height_dataset.zip"  # Path to zip file
    extract_to = "data/"  

    # Check if the file exists
    if os.path.exists(zip_path):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete!")
    else:
        print(f"File {zip_path} does not exist.")

if __name__ == "__main__":
    download_and_extract_data()
