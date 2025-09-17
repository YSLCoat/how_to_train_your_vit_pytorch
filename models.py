import torch.nn as nn
import torchvision.models as torch_models
from vit_pytorch import ViT

MODEL_REGISTRY = {
    "resnet18": torch_models.resnet18,
    "resnet50": torch_models.resnet50,
    "vgg16": torch_models.vgg16,
    "lucidrain_vit": ViT,
}


def create_model(args):
    model_constructor = MODEL_REGISTRY.get(args.arch)

    if not model_constructor:
        raise ValueError(f"Model {args.arch} not supported.")

    if args.arch == "lucidrain_vit":
        model = model_constructor(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            dim=args.dim,
            depth=args.depth,
            heads=args.heads,
            mlp_dim=args.mlp_dim,
            dropout=args.dropout,
            emb_dropout=args.emb_dropout,
        )
    else:
        model = model_constructor(pretrained=args.pretrained)

    return model
