# Deepfake_ensembling
# DeepFake Ensemble Detection

This repository demonstrates an approach for detecting deepfake images using state-of-the-art deep learning models. The project is divided into two main parts:  
1. **DeepFake Image Classification with Vision Transformer (ViT):** A ViT-based model is fine-tuned on a deepfake image dataset to classify images as **real** or **fake**.  
2. **Ensemble Method:** An ensemble approach is implemented by combining the ViT model with a ResNet-based model to improve detection performance.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
  - [Vision Transformer (ViT)](#vision-transformer-vit)
  - [ResNet](#resnet)
  - [Ensemble Approach](#ensemble-approach)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
  
## Overview

Deepfake images continue to be a significant challenge in verifying digital media authenticity. This project tackles the problem by fine-tuning a Vision Transformer (ViT) model for binary classification (real vs. fake) and further improves performance using an ensemble method that averages predictions from both a ViT and a ResNet model. The ensemble method leverages the strengths of both architectures to achieve higher accuracy and robustness.

## Dataset

The dataset is structured as follows:
deepfake-image-detection/
    ├── train/ 
        ├── real/ 
        ├──fake/ 
    ├──val/ 
        ├── real/ 
        ├── fake/
Each folder contains images categorized as either **real** or **fake**. Standard preprocessing steps (resizing to 224×224, normalization, etc.) are applied to prepare the images for model training.

## Models

### Vision Transformer (ViT)

- **Architecture:**  
  The ViT model divides input images into patches and processes them with transformer layers, leveraging self-attention to capture global context.
- **Implementation:**  
  A pretrained ViT model is fine-tuned by replacing its classification head to output two classes. This model forms the backbone for our deepfake detection task.

### ResNet

- **Architecture:**  
  ResNet (Residual Network) utilizes skip connections to mitigate the vanishing gradient problem, making it highly effective for image classification.
- **Implementation:**  
  A ResNet-18 model pretrained on ImageNet is adapted for binary classification by replacing the final fully connected layer.

### Ensemble Approach

- **Methodology:**  
  The ensemble combines predictions from both the ViT and ResNet models by averaging their softmax probabilities.
- **Benefits:**  
  This method helps to reduce individual model biases and improves overall detection accuracy by leveraging complementary features learned by each architecture.
  



