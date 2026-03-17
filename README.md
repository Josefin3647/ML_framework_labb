# Project Name

## Description


## Usage

## Project structure
## Dataset

## Dataset

This project uses the CIFAR-10 dataset, a widely used benchmark for image classification tasks.
It consists of 60,000 color images of size 32×32 pixels, divided into 10 classes.

The dataset is split into:

* 50,000 training images
* 10,000 test images

Images are automatically downloaded and stored in the `data/` directory.

## Data Preprocessing and Augmentation

The training data is augmented using random transformations to improve generalization.
The following transformations are applied:

* Random horizontal flip (p=0.5)
* Random crop (32x32) with padding=4
* Conversion to tensor
* Normalization using dataset-specific mean and standard deviation

Augmentation is applied on-the-fly during training, but not during evaluation. 

## Acknowledgments

This project was developed with assistance from ChatGPT, which was used to help generate and refine parts of the Python code.

