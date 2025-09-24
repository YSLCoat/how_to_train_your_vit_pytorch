import argparse
import pathlib
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torchvision.transforms as transforms
from torchsummary import summary
import torch.nn.functional as F

from vit_pytorch.recorder import Recorder

from utils.data_utils import build_data_loaders
from models import create_model

image_transforms = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
)


def class_idx_to_label(file_path: str, row_index: int) -> str:
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
        return lines[row_index]


def main(args):
    state_dict = torch.load(args.model_path, weights_only=False)
    model = create_model(state_dict['args']).to('cuda')
    model.load_state_dict(state_dict['state_dict'])
    summary(model.to('cuda'), (3, 224, 224))

    v = Recorder(model)
    image = Image.open(args.input_image_path).convert("RGB")
    transformed_image = image_transforms(image)

    plt.imshow(image)
    plt.show()
    
    with torch.no_grad():
        logits, attention_scores = v(transformed_image.unsqueeze(0).to('cuda'))

    preds = torch.softmax(logits, dim=1)

    predicted_class_idx = torch.argmax(preds, dim=1)
    predicted_class_score = preds[:,predicted_class_idx.item()]
    print(predicted_class_score)
    predicted_class = class_idx_to_label(args.synset_mapping_path, predicted_class_idx.item())
    plt.imshow(image)
    plt.title(predicted_class)
    plt.show()

    print(attention_scores.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on input image.")

    parser.add_argument(
        "--model_path",
        type=pathlib.Path,
        required=True,
        help="if specified applies mixup augmentation for inputs",
    )
    parser.add_argument(
        "--synset_mapping_path",
        type=pathlib.Path,
        help="Path to LOC_synset_mapping.txt file"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_image_path",
        type=pathlib.Path,
        help="Path to image used for model inference.",
    )
    input_group.add_argument(
        "--input_image_folder_path",
        type=pathlib.Path,
        help="Path to folder containing images to be used for inference.",
    )

    args = parser.parse_args()
    main(args)