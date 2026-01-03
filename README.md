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
* **Distributed Training (Training on muiltiple GPUs)**: Support for Distributed Data Parallel (DDP) and multi-processing on single or multiple nodes.
* **Experiment Tracking**: Includes utilities for saving checkpoints, inspecting training history (`inspect_history.py`), and running inference (`predict.py`).

## Data
Imagenet can be downloaded from here: https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data . Its 167GB so it takes some time to download. After you download the data you need to unzip it as well, this will also take some time. The training data is already structured as the dataclass in data_utils.py expects it to be, however the images in the validation folder is not grouped by their classes yet. To group the images in the validation folder navigate to the validation folder in your terminal and run `wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash` in your terminal. This will automatically sort the data. 

## Quick Start
You can use the provided shell scripts to start a default training run:
```bash
# Linux / Mac
./default_train.sh

# Windows
./default_train.bat
```
Alternatively: `python train.py --data "path/to/rootdir/of/data" --arch "lucidrain_vit" --epochs 36 --batch-size 256 --output_parent_dir "path/to/where/you/wanna/store/trained/model" --mixup `

## Results from run with default arguments
Training with 36 epochs should give results like these:
![Loss Curve](figures/loss_curves.png)
with training time on an RTX 5090 being roughly 1 day.