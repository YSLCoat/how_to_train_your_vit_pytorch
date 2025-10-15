#!/bin/sh
# This is a comment!
n_epochs=90
batch_size=256
model_architecture="lucidrain_vit"
data_path="/home/yslcoat/data/imagenet1k"
model_save_dir="/home/yslcoat/trained_models"

systemd-inhibit --what=sleep python train.py \
    --data "$data_path" \
    --arch "$model_architecture" \
    --epochs "$n_epochs" \
    --batch-size "$batch_size" \
    --output_parent_dir "$model_save_dir" \
    --mixup