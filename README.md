# IPEO_Canopy_height
Forest Canopy height regression from EPFL Image Processing for Earth Observation Course

## Project description

# Task
Vegetation height is a fundamental variable necessary for estimating carbon fluxes, understanding biodiversity, ecosystems services and many other applications. In this project, you will predict per-pixel vegetation height from multispectral Sentinel-2 images.
# Data
A total of 9852 images (and the corresponding masks) will be used in this project. You can see
an example of both in Figure 1 (see project proposal). Each pair of image and mask is named with a unique id. In the data folder, you will find a CSV file where each id is assigned to either the train, validation or the test set.

• Images. Patches of Sentinel-2 multi-spectral images, all bands upsampled to a common 10m
resolution. Each patch is of size 32 × 32.
• Labels. The labels are in the form of segmentation masks. There is one segmentation mask for
each image with the identical size 32 × 32 pixels. The pixel values in the mask correspond to the
canopy height in meters as modeled by [1]. Values of 255 correspond to ’nodata’ values.

Data can be accessed at https://drive.google.com/file/d/1iRQDJ4qCmUGrLyjgzeFcvxir90IlYohr/view?usp=drive_link

# Challenges
• You need to perform per-pixel regression (predict a continuous output for every pixel); your
model architecture and loss function need to reflect that.
• Sentinel-2 images have 12 input bands; most standard deep learning models have been developed
with natural 3-band images in mind; you may need to adapt an existing architecture to serve your
purposes

# References 
[1] Lang, N., Kalischek, N., Armston, J., Schindler, K., Dubayah, R. and Wegner, J.D., 2022.
Global canopy height regression and uncertainty estimation from GEDI LIDAR waveforms with deep
ensembles. Remote sensing of environment, 268, p.112760.

## Project state

This project is being developped by EPFL students. Due date : 15/01/2025, on Moodle.

# Requirements
For both BYOP and deep learning projects, your submission should contain:
1. the report (max 10 pages)
2. the code (url of public Github repository or .zip file)
3. a jupyter notebook named inference.ipynb that
a. loads at least one image/sample from the test set
b. loads trained parameters from the best model you trained
c. runs inference (i.e. applies the model) on one image from the test set
d. displays the predicJons for this image
4. the test image/sample used in inference.ipynb
5. the trained parameters of the model used in inference.ipynb: file or download link
(Google Drive, Switch Filesender, etc.)
6. a requirement file named environment.yml (documentaJon) that lists all the python
packages you used.


The jupyter notebook can use functions defined elsewhere in your code. Please make sure it
runs out of the box, ideally by downloading and testing your submission on a new
computer/environment.


## How to use this repository?
First, clone it in your computer in your IPEO project folder.

Each time you want to change something:
- make sure to make a PULL REQUEST BEFORE BEGGINIG TO WORK
- if you want to make some special improvement, create a branch
- PULL and COMMIT your changes when done (Remember to put your changes in the commit files)


Don't hesitate to create "issues" on GitHub. And to help me with Git because I am new with it.

