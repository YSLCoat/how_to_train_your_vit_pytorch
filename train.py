import os
import random
import time
import warnings
import logging
import pathlib
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Subset

from utils.training_utils import (
    save_checkpoint,
    load_checkpoint,
)
from model_utils import configure_multi_gpu_model
from utils.data_utils import build_data_loaders
from input_parser import build_config
from metrics_utils import MetricsEngine
from models import create_model

logger = logging.getLogger()
best_acc1 = 0


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    logging.info(f"Using device: {device}")

    if device.type == "cuda":
        ngpus_per_node = torch.accelerator.device_count()
        if ngpus_per_node == 1 and args.dist_backend == "nccl":
            warnings.warn(
                "nccl backend >=2.5 requires GPU count>1, see https://github.com/NVIDIA/nccl/issues/103 perhaps use 'gloo'"
            )
    else:
        ngpus_per_node = 1

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # TODO: Improve implementation of training_id generation.
    unique_training_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{args.arch}_{unique_training_id}"
    global best_acc1
    args.gpu = gpu

    use_accel = not args.no_accel and torch.accelerator.is_available()

    if use_accel:
        if args.gpu is not None:
            torch.accelerator.set_device_index(args.gpu)
        device = torch.accelerator.current_accelerator()
    else:
        device = torch.device("cpu")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    logging.info("=> creating model '{}'".format(args.arch))
    model = create_model(args)

    if not use_accel:
        logging.info("using CPU, this will be slow")
    else:
        configure_multi_gpu_model(args, model, device, ngpus_per_node)

    train_loader, val_loader, train_sampler, _ = build_data_loaders(args)

    criterion = nn.CrossEntropyLoss().to(device)

    total_steps = len(train_loader) * args.epochs

    optimizer = torch.optim.AdamW(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )

    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - args.warmup_period, eta_min=1e-6
    )

    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=args.warmup_period
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[args.warmup_period],
    )

    metrics_engine = MetricsEngine(use_accel)

    if args.resume:
        load_checkpoint(args, device, model, optimizer, scheduler, metrics_engine)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            metrics_engine,
            epoch,
            device,
            args,
        )
        metrics_engine.update_epoch()

        acc1 = validate(val_loader, model, criterion, metrics_engine, args, epoch)
        metrics_engine.update_epoch()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "args": args,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "batch_history": metrics_engine.batch_history,
                    "epoch_history": metrics_engine.epoch_history,
                },
                is_best,
                pathlib.Path(args.output_parent_dir, folder_name),
            )


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    metrics_engine,
    epoch,
    device,
    args,
):
    metrics_engine.set_mode("train")
    metrics_engine.reset_metrics()
    metrics_engine.configure_progress_meter(len(train_loader), epoch)

    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        data_time = time.time() - end

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        # select dominant label for accuracy calculation when using mixup
        if args.mixup:
            target = target.argmax(dim=1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()
        metrics_engine.update_batch(
            data_time, loss.item(), time.time() - end, output, target
        )
        end = time.time()

        if i % args.print_freq == 0:
            metrics_engine.progress.display(i + 1)


def validate(val_loader, model, criterion, metrics_engine, args, epoch=None):
    metrics_engine.set_mode("validate")
    metrics_engine.reset_metrics()
    metrics_engine.configure_progress_meter(
        len(val_loader)
        + (
            args.distributed
            and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))
        ),
        epoch,
    )

    use_accel = not args.no_accel and torch.accelerator.is_available()

    def run_validate(loader, base_progress=0):

        if use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(loader):
                data_time = time.time() - end
                i = base_progress + i
                if use_accel:
                    if args.gpu is not None and device.type == "cuda":
                        torch.accelerator.set_device_index(args.gpu)
                        images = images.cuda(args.gpu, non_blocking=True)
                        target = target.cuda(args.gpu, non_blocking=True)
                    else:
                        images = images.to(device)
                        target = target.to(device)

                output = model(images)
                loss = criterion(output, target)

                metrics_engine.update_batch(
                    data_time, loss.item(), time.time() - end, output, target
                )

                end = time.time()

                if i % args.print_freq == 0:
                    metrics_engine.progress.display(i + 1)

    model.eval()

    run_validate(val_loader)
    if args.distributed:
        metrics_engine.top1.all_reduce()
        metrics_engine.top5.all_reduce()

    if args.distributed and (
        len(val_loader.sampler) * args.world_size < len(val_loader.dataset)
    ):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    metrics_engine.progress.display_summary()

    return metrics_engine.top1.avg


if __name__ == "__main__":
    args = build_config()
    main(args)
