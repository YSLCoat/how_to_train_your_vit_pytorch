import shutil
import os
import logging
import pathlib
from enum import Enum

import torch
import torch.distributed as dist


def save_checkpoint(state, is_best, path, filename="checkpoint.pth.tar"):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(state, pathlib.Path(path, filename))
    if is_best:
        shutil.copyfile(pathlib.Path(path, filename), pathlib.Path(path, "model_best.pth.tar"))


def load_checkpoint(args, device, model, optimizer, scheduler, metrics_engine) -> None:
    if os.path.isfile(args.resume):
        logging.info("=> loading checkpoint '{}'".format(args.resume))
        if args.gpu is None:
            checkpoint = torch.load(args.resume, weights_only=False)
        else:
            loc = f"{device.type}:{args.gpu}"
            checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint["epoch"]
        best_acc1 = checkpoint["best_acc1"]
        if args.gpu is not None:
            best_acc1 = best_acc1.to(args.gpu)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        metrics_engine.batch_history = checkpoint["batch_history"]
        metrics_engine.epoch_history = checkpoint["epoch_history"]
        logging.info(
            "=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint["epoch"]
            )
        )
    else:
        logging.info("=> no checkpoint found at '{}'".format(args.resume))
