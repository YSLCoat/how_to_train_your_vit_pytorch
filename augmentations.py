import torch
import torchvision
from torchvision.transforms import v2
from torch.utils.data import default_collate

AUGMENTATIONS = {}

# normalize = v2.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])


# default_train = v2.Compose([
#     v2.RandomResizedCrop(size=(224, 224), antialias=True),
#     v2.RandomHorizontalFlip(p=0.5),
#     v2.ToTensor(),
#     normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


# default_val = v2.Compose([
#                 v2.Resize(224),
#                 v2.CenterCrop(224),
#                 v2.ToTensor(),
#                 normalize,
#             ])

# vit_augment_train = v2.Compose([
#                 v2.Resize(224),
#                 v2.RandAugment(),
#                 v2.ToTensor(),
#                 normalize,
#             ])


class MixUpCollator:
    def __init__(self, num_classes):
        self.mixup = v2.MixUp(num_classes=num_classes)

    def __call__(self, batch):
        return self.mixup(*default_collate(batch))
