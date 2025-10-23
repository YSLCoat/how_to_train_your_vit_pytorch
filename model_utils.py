from dataclasses import dataclass

import torch
import torch.nn as nn
import torchvision.models as torch_models
from models import LucidrainViT

MODEL_REGISTRY = {
    "resnet18": torch_models.resnet18,
    "resnet50": torch_models.resnet50,
    "vgg16": torch_models.vgg16,
    "lucidrain_vit": LucidrainViT,
}


def create_model(args):
    learning_tasks_list = create_learning_tasks(args.learning_tasks.copy())
    model_constructor = MODEL_REGISTRY.get(args.arch)

    if not model_constructor:
        raise ValueError(f"Model {args.arch} not supported.")

    if args.arch == "lucidrain_vit":
        model = model_constructor(
            learning_tasks=learning_tasks_list,
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



def configure_training_device(args):
    use_accel = not args.no_accel and torch.accelerator.is_available()
    if use_accel:
        if args.gpu is not None:
            torch.accelerator.set_device_index(args.gpu)
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    return device


def configure_multi_gpu_model(args, model, device, ngpus_per_node):
    if args.distributed:
        if device.type == "cuda":
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(device)
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[args.gpu]
                )
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif device.type == "cuda":
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
    else:
        model.to(device)


class LearningTask:
    def __init__(self, name: str):
        self.name = name

    def create_criterion(self) -> nn.Module:
        raise NotImplementedError


class ClassificationLearningTask(LearningTask):
    def __init__(self, name: str, n_classes: int):
        super().__init__(name)
        self.output_layer_shape = n_classes

    def create_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss()


class ObjectDetectionLearningTask(LearningTask):
    def __init__(self, name: str, **kwargs):
        super().__init__(name)
        raise NotImplementedError
    
    def create_criterion(self) -> nn.Module:
        raise NotImplementedError


TASK_REGISTRY = {
    "classification": ClassificationLearningTask,
    "object_detection": ObjectDetectionLearningTask,
    # "semantic_segmentation": SemanticSegmentationLearningTask,
    # "instance_segmentation": InstanceSegmentationLearningTask,
}


def create_learning_tasks(task_configs: list[dict]) -> list[LearningTask]:
    tasks = []
    for task_config in task_configs:
        task_type = task_config.pop("learning_task")
        TaskClass = TASK_REGISTRY.get(task_type)
        if not TaskClass:
            raise ValueError(f"Task type '{task_type}' not supported. Supported learning tasks: {TASK_REGISTRY.keys()}.")
        
        tasks.append(TaskClass(**task_config))
    return tasks
    
