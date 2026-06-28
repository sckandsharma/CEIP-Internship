# Autoencoder for Image Denoising - MNIST

## Overview

This project implements a Convolutional Autoencoder to remove noise from handwritten digit images using the MNIST dataset. The model is trained to take noisy images as input and reconstruct clean images as output.

## Problem Statement

Build a deep learning model that can remove noise from images using an autoencoder on the MNIST dataset.

## Dataset

The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels.

- Training samples: 60,000
- Testing samples: 10,000

Dataset source: https://www.kaggle.com/datasets/jidhumohan/mnist-png

## Approach

1. Load the MNIST dataset using torchvision
2. Add Gaussian noise to clean images to create corrupted input
3. Train the autoencoder to reconstruct clean images from noisy input
4. Evaluate by visually comparing original, noisy, and denoised images

## Model Architecture

The autoencoder consists of two main components:

**Encoder**
- Conv2d(1, 32, kernel_size=3, padding=1) -> ReLU -> MaxPool2d
- Conv2d(32, 16, kernel_size=3, padding=1) -> ReLU -> MaxPool2d
- Output: compressed representation of size (16, 7, 7)

**Decoder**
- ConvTranspose2d(16, 32, kernel_size=2, stride=2) -> ReLU
- ConvTranspose2d(32, 1, kernel_size=2, stride=2) -> Sigmoid
- Output: reconstructed image of size (1, 28, 28)

## Training Details

- Loss Function: MSELoss (pixel-level reconstruction task)
- Optimizer: Adam (learning rate = 0.001)
- Epochs: 10
- Batch Size: 64

MSELoss is used because this is a regression task where we compare pixel values between the output image and the target clean image.

## Requirements

torch
torchvision
numpy
matplotlib

Install dependencies:

pip install torch torchvision numpy matplotlib

## How to Run

1. Clone the repository
2. Install the required libraries
3. Open the notebook Autoencoder_Image_Denoising.ipynb
4. Run all cells sequentially

## Results

The model produces three outputs for comparison:
- Original clean image
- Noisy image (model input)
- Denoised image (model output)

The training loss decreases consistently over 10 epochs, indicating the model is successfully learning to remove noise and reconstruct the original digits.

## Project Structure

MNIST_Autoencoder_Assignment/
    Autoencoder_Image_Denoising.ipynb
    denoising_autoencoder.pth
    README.md

## References

- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- PyTorch Documentation: https://pytorch.org/docs/stable/index.html
- Reference Repository: https://github.com/NvsYashwanth/MNIST-Autoecncoder
