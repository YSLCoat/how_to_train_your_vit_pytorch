import os
import pathlib
import logging
from PIL import Image
from typing import Tuple, Dict, List
import xml.etree.ElementTree as ET

import torch
import torch.utils.data
from torch.utils.data import (
    Dataset,
    default_collate
)
import torch.utils.data.distributed
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms import v2


logger = logging.getLogger()

class ImageNetDataset(Dataset):
    """
    Custom imagenet dataset class. Expects the following file structure
    root_dir/
    ├── Data/
    │   └── CLS-LOC /
    │       ├── train /
    |       |   └── class_folders /
    |       |       └── filename.JPEG 
    |       |
    │       └── val /...
    │   
    └── Annotations/
        └── CLS-LOC /
            ├── train /
            |   └── class_folders /
            |       └── filename.xml 
            └── val /...
    """
    # img path: root folder -> Data -> CLS-LOC -> test/train/val -> class_folders -> filename.JPEG
    # annotation path: root folder -> Annotations -> CLS-LOC -> train/val -> class_folders -> filename.xml
    def __init__(self, root_dir: str, partition:str='train', transforms=None, object_detection=False) -> None:
        self.img_dir = pathlib.Path(os.path.join(root_dir, 'Data', 'CLS-LOC', partition))
        self.object_detection = object_detection
        if object_detection:
            self.annotation_dir = pathlib.Path(os.path.join(root_dir, 'Annotations', 'CLS-LOC', partition))
            self.annotation_paths = [f for f in self.annotation_dir.rglob('*') if f.suffix.lower() == '.xml'] 
            self.img_paths = []

            for path in self.annotation_paths:
                path_parts = list(path.parts)
                data_index = path_parts.index('Annotations')
                path_parts[data_index] = 'Data'
                img_base_path = pathlib.Path(*path_parts)
                self.img_paths.append(img_base_path.with_suffix('.JPEG'))                
        else:
            self.img_paths = [f for f in self.img_dir.rglob('*') if f.suffix == '.JPEG']

        self.readable_classes_dict = extract_readable_imagenet_labels(os.path.join(root_dir, 'LOC_synset_mapping.txt'))
        self.transforms = transforms
        self.classes, self.class_to_idx = find_classes(self.img_dir, self.readable_classes_dict)

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.img_paths[index]
        return Image.open(image_path).convert('RGB') 
    
    def load_bounding_box_coords(self, index:int, img_size:Tuple) -> torch.Tensor:
        annotation_path = self.annotation_paths[index]
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        bndbox = root.find(".//bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)

        return torchvision.tv_tensors.BoundingBoxes([[xmin, ymin, xmax, ymax]], format='XYXY', canvas_size=img_size)
        
    
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.img_paths)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name = self.img_paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        readable_class_name = self.readable_classes_dict[class_name]
        class_idx = self.class_to_idx[readable_class_name]

        if self.object_detection:
            H, W = img.height, img.width
            bndbox_coords_tensor = self.load_bounding_box_coords(index, (H, W))

            if self.transforms:
                return self.transforms({'image': img, 'boxes': bndbox_coords_tensor, 'labels': torch.tensor([class_idx])})
            else:
                return img, bndbox_coords_tensor, class_idx
        else:
            if self.transforms:
                return self.transforms(img), class_idx
            else:
                return img, class_idx
        

class MixUpCollator:
    def __init__(self, num_classes):
        self.mixup = v2.MixUp(num_classes=num_classes)

    def __call__(self, batch):
        return self.mixup(*default_collate(batch))
    
        
def extract_readable_imagenet_labels(file_path: os.path) -> dict:
    """
    Helper function for storing imagenet human read-able 
    class mappings. Mapping downloaded from 
    https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a 
    """
    class_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            words = line.strip().split()
            class_dict[words[0]] = words[1].rstrip(',') # Incase there are several readable labels which are comma separated. 

    return class_dict


def find_classes(directory: str, readable_classes_dict: dict) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    readable_classes = [readable_classes_dict.get(key) for key in classes]
    
    if not readable_classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(readable_classes)}
    return readable_classes, class_to_idx

def build_data_loaders(args):
    if args.dummy:
        logging.info("=> Dummy data is used!")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        logging.info("loading data")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

        train_dataset = ImageNetDataset(
            args.data, 'train',
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandAugment(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = ImageNetDataset(
            args.data, 'val',
            transforms.Compose([
                transforms.Resize((224,224)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        val_sampler = None

    collate_fn = MixUpCollator(num_classes=args.num_classes) if args.mixup else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)
    
    return train_loader, val_loader, train_sampler, val_sampler
    