import argparse

from models import MODEL_REGISTRY


def build_config():
    model_names = sorted(MODEL_REGISTRY.keys())

    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument(
        "--data",
        metavar="DIR",
        nargs="?",
        default="imagenet",
        help="path to dataset (default: imagenet)",
    )
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names) + " (default: resnet18)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=5e-4,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--warmup_period",
        type=int,
        default=10000,
        help="number of warmup steps for lr before scheduler",
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=0.05,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://224.66.41.62:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument("--no-accel", action="store_true", help="disables accelerator")
    parser.add_argument(
        "--multiprocessing-distributed",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "--dummy", action="store_true", help="use fake data to benchmark"
    )

    # Model configs
    # ViT
    parser.add_argument(
        "--image_size", type=int, default=224, help="size of input image to vit."
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
        help="size of patches. image_size must be divisible by patch size.",
    )
    parser.add_argument(
        "--num_classes", type=int, default=1000, help="number of classes in dataset."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=1024,
        help="last dimension of output tensor after linear transformation.",
    )
    parser.add_argument(
        "--depth", type=int, default=6, help="number of transformer blocks."
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=16,
        help="number of heads in multi-head attention layer.",
    )
    parser.add_argument(
        "--mlp_dim",
        type=int,
        default=2048,
        help="dimension of mlp (feedforward) layer.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="dropout rate between [0, 1]"
    )
    parser.add_argument(
        "--emb_dropout", type=float, default=0.1, help="dropout rate between [0, 1]"
    )

    # Augmentations
    parser.add_argument(
        "--mixup",
        action="store_true",
        help="if specified applies mixup augmentation for inputs",
    )

    args = parser.parse_args()
    return args
