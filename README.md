# How To Train Your ViT - PyTorch

A PyTorch implementation of the training framework presented in [**"How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers"**](https://arxiv.org/abs/2106.10270).

This repository extends the official [PyTorch ImageNet training example](https://github.com/pytorch/examples/tree/main/imagenet) by integrating the specific regularization, data augmentation, and optimization techniques required to train Vision Transformers.

## Key Features

* **Model Optimization**: Implements the specific training setup described in the paper, including:
    * **Optimizer Setup**
    * **Learning Rate Schedule**: Linear Warmup followed by Cosine Annealing.
* **Augmentation**
    * **Mixup**
    * **RandAugment**
* **Distributed Training**: Robust support for Distributed Data Parallel (DDP) and multi-processing on single or multiple nodes.
* **Experiment Tracking**: Includes utilities for saving checkpoints, inspecting training history (`inspect_history.py`), and running inference (`predict.py`).

## Usage

### Quick Start
You can use the provided shell scripts to start a default training run:
```bash
# Linux / Mac
./default_train.sh

# Windows
./default_train.bat
