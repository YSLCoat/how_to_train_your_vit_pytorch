import os
import random
import time
import warnings
import logging
import argparse

from metrics import accuracy

import torch
from torch.utils.data import Subset
from utils.training_utils import (
    save_checkpoint,
    AverageMeter,
    ProgressMeter,
    Summary
)

class Trainer():
    def __init__(self):
        pass


    def train_epoch(self, train_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, epoch: int, device, args: argparse.ArgumentParser):
        use_accel = not args.no_accel and torch.accelerator.is_available()

        batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
        data_time = AverageMeter('Data', use_accel, ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
        top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.NONE)
        top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.NONE)
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

        model.train()

        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            data_time.update(time.time() - end)

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            # select dominant label for accuracy calculation when using mixup
            if args.mixup:
                target = target.argmax(dim=1)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i + 1)


    def validate(self, val_loader: torch.utils.data.DataLoader, model: torch.nn.Module, criterion: torch.nn.Module, args: argparse.ArgumentParser):
        use_accel = not args.no_accel and torch.accelerator.is_available()

        def run_validate(loader, base_progress=0):

            if use_accel:
                device = torch.accelerator.current_accelerator()
            else:
                device = torch.device("cpu")

            with torch.no_grad():
                end = time.time()
                for i, (images, target) in enumerate(loader):
                    i = base_progress + i
                    if use_accel:
                        if args.gpu is not None and device.type=='cuda':
                            torch.accelerator.set_device_index(args.gpu)
                            images = images.cuda(args.gpu, non_blocking=True)
                            target = target.cuda(args.gpu, non_blocking=True)
                        else:
                            images = images.to(device)
                            target = target.to(device)

                    output = model(images)
                    loss = criterion(output, target)

                    acc1, acc5 = accuracy(output, target, topk=(1, 5))
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0], images.size(0))
                    top5.update(acc5[0], images.size(0))

                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % args.print_freq == 0:
                        progress.display(i + 1)

        batch_time = AverageMeter('Time', use_accel, ':6.3f', Summary.NONE)
        losses = AverageMeter('Loss', use_accel, ':.4e', Summary.NONE)
        top1 = AverageMeter('Acc@1', use_accel, ':6.2f', Summary.AVERAGE)
        top5 = AverageMeter('Acc@5', use_accel, ':6.2f', Summary.AVERAGE)
        progress = ProgressMeter(
            len(val_loader) + (args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset))),
            [batch_time, losses, top1, top5],
            prefix='Test: ')

        model.eval()

        run_validate(val_loader)
        if args.distributed:
            top1.all_reduce()
            top5.all_reduce()

        if args.distributed and (len(val_loader.sampler) * args.world_size < len(val_loader.dataset)):
            aux_val_dataset = Subset(val_loader.dataset,
                                    range(len(val_loader.sampler) * args.world_size, len(val_loader.dataset)))
            aux_val_loader = torch.utils.data.DataLoader(
                aux_val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            run_validate(aux_val_loader, len(val_loader))

        progress.display_summary()

        return top1.avg
    
    def train(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            if self.args.distributed:
                self.train_sampler.set_epoch(epoch)

            self.train_epoch(self.train_loader, self.model, self.criterion, self.optimizer, self.scheduler, self.epoch, self.device, self.args)

            acc1 = self.validate(self.val_loader, self.model, self.criterion, self.args)
            
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                    and self.args.rank % self.ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': self.model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer' : self.optimizer.state_dict(),
                    'scheduler' : self.scheduler.state_dict()
                }, is_best)