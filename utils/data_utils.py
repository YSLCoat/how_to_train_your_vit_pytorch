import os
import pathlib
import logging
from PIL import Image
from typing import Tuple, List, Dict, Any, Set
import xml.etree.ElementTree as ET

import torch
import torch.utils.data
from torch.utils.data import Dataset, default_collate
import torch.utils.data.distributed
from torchvision.tv_tensors import BoundingBoxes
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
    
    SUPPORTED_TASKS: Set[str] = {"classification", "object_detection"}

    def __init__(
        self,
        root_dir: str,
        learning_tasks: List[Dict[str, Any]],
        partition: str = "train",
        transforms=None,
    ) -> None:
        self.learning_tasks_config = learning_tasks
        self.img_dir = pathlib.Path(
            os.path.join(root_dir, "ILSVRC", "Data", "CLS-LOC", partition) 
        )

        self.task_types = set()
        self.tasks_by_name = {}
        
        for task_config in self.learning_tasks_config:
            task_type = task_config['learning_task']
            task_name = task_config['name']
            
            if task_type not in self.SUPPORTED_TASKS:
                raise ValueError(
                    f"Dataset {self.__class__.__name__} does not support task type '{task_type}'. "
                    f"Supported tasks are: {self.SUPPORTED_TASKS}"
                )
            
            if task_name in self.tasks_by_name:
                raise ValueError(f"Duplicate task name '{task_name}' found. Task names must be unique.")
                
            self.task_types.add(task_type)
            self.tasks_by_name[task_name] = task_config

        if 'object_detection' in self.task_types:
            self.annotation_dir = pathlib.Path(
                os.path.join(root_dir, "Annotations", "CLS-LOC", partition)
            )
            self.annotation_paths = [
                f for f in self.annotation_dir.rglob("*") if f.suffix.lower() == ".xml"
            ]
            self.img_paths = []

            for path in self.annotation_paths:
                path_parts = list(path.parts)
                try:
                    data_index = path_parts.index("Annotations")
                    path_parts[data_index] = "Data"
                    img_base_path = pathlib.Path(*path_parts)
                    self.img_paths.append(img_base_path.with_suffix(".JPEG"))
                except ValueError:
                    print(f"Warning: Could not derive image path from annotation path: {path}")
        else:
            self.img_paths = [
                f for f in self.img_dir.rglob("*") if f.suffix.lower() == ".jpeg"
            ]
            self.annotation_paths = None

        self.readable_classes_dict = extract_readable_imagenet_labels(
            os.path.join(root_dir, "LOC_synset_mapping.txt")
        )
        self.transforms = transforms
        self.classes, self.class_to_idx = find_classes(
            self.img_dir, self.readable_classes_dict
        )

    def load_image(self, index: int) -> Image.Image:
        image_path = self.img_paths[index]
        return Image.open(image_path).convert("RGB")

    def load_bounding_box_coords(self, index: int, img_size: Tuple) -> BoundingBoxes:
        if self.annotation_paths is None:
            raise RuntimeError("Called load_bounding_box_coords but no annotation paths were loaded.")
            
        annotation_path = self.annotation_paths[index]
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        
        boxes = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox:
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                boxes.append([xmin, ymin, xmax, ymax])

        if not boxes:
            return BoundingBoxes(
                [], format="XYXY", canvas_size=img_size
            )

        return BoundingBoxes(
            boxes, format="XYXY", canvas_size=img_size
        )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[Any, Dict[str, Any]]:
        img = self.load_image(index)
        
        class_name = self.img_paths[
            index
        ].parent.name
        readable_class_name = self.readable_classes_dict.get(class_name, class_name)
        class_idx = self.class_to_idx[readable_class_name]

        target_dict = {}

        if 'object_detection' not in self.task_types:            
            for task_name, task_config in self.tasks_by_name.items():
                if task_config['learning_task'] == 'classification':
                    target_dict[task_name] = class_idx
                    
            if self.transforms:
                img = self.transforms(img)
                
            return img, target_dict

        else:
            H, W = img.height, img.width
            bndbox_coords_tensor = self.load_bounding_box_coords(index, (H, W))
            
            box_labels = torch.full((len(bndbox_coords_tensor),), 
                                     class_idx, 
                                     dtype=torch.int64)

            for task_name, task_config in self.tasks_by_name.items():
                task_type = task_config['learning_task']
                
                if task_type == 'classification':
                    target_dict[task_name] = class_idx
                    
                elif task_type == 'object_detection':
                    target_dict[task_name] = {
                        'boxes': bndbox_coords_tensor,
                        'labels': box_labels
                    }

            if self.transforms:
                sample = {
                    'image': img,
                    'boxes': bndbox_coords_tensor,
                    'labels': box_labels
                }
                
                transformed_sample = self.transforms(sample)
                transformed_img = transformed_sample['image']
                
                for task_name, task_config in self.tasks_by_name.items():
                    if task_config['learning_task'] == 'object_detection':
                        target_dict[task_name] = {
                            'boxes': transformed_sample['boxes'],
                            'labels': transformed_sample['labels']
                        }
                
                return transformed_img, target_dict
            
            else:
                return img, target_dict
            

class FakeDataset(datasets.FakeData):
    def __init__(self, *args, **kwargs):
            # TODO: Add support for learning tasks for classification, obj detection, semantic and instance segmentation.
            raise NotImplementedError
    


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

    with open(file_path, "r") as file:
        for line in file:
            words = line.strip().split()
            class_dict[words[0]] = words[1].rstrip(
                ","
            )  # Incase there are several readable labels which are comma separated.

    return class_dict


def find_classes(
    directory: str, readable_classes_dict: dict
) -> Tuple[List[str], Dict[str, int]]:
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
        train_dataset = datasets.FakeData(
            1281167, (3, 224, 224), 1000, transforms.ToTensor()
        )
        val_dataset = datasets.FakeData(
            50000, (3, 224, 224), 1000, transforms.ToTensor()
        )
    else:
        logging.info("loading data")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = ImageNetDataset(
            args.data,
            args.learning_tasks,
            "train",
            transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandAugment(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        val_dataset = ImageNetDataset(
            args.data,
            args.learning_tasks,
            "val",
            transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=True
        )
    else:
        train_sampler = None
        val_sampler = None

    collate_fn = MixUpCollator(num_classes=args.num_classes) if args.mixup else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
    )

    return train_loader, val_loader, train_sampler, val_sampler
