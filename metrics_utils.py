from enum import Enum
from collections import defaultdict

import torch
import torch.distributed as dist

from metrics import accuracy


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, use_accel, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.use_accel = use_accel
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if self.use_accel:
            device = torch.accelerator.current_accelerator()
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class MetricsEngine:
    def __init__(self, use_accel, args):
        self.batch_time = AverageMeter("Time", use_accel, ":6.3f", Summary.NONE)
        self.data_time = AverageMeter("Data", use_accel, ":6.3f", Summary.NONE)
        self.losses = AverageMeter("Loss", use_accel, ":.4e", Summary.NONE)
        self.top1 = AverageMeter("Acc@1", use_accel, ":6.2f", Summary.NONE)
        self.top5 = AverageMeter("Acc@5", use_accel, ":6.2f", Summary.NONE)

        self.batch_history = defaultdict(list)
        self.epoch_history = defaultdict(list)

        self.mode = None

    def configure_progress_meter(self, n_samples, epoch=None):
        if self.mode == "train":
            prefix = "Training Epoch: [{}]".format(epoch)
        elif self.mode == "validate":
            prefix = "Validate Epoch: [{}]".format(epoch)
        else:
            prefix = "Test: "

        self.progress = ProgressMeter(
            n_samples,
            [self.batch_time, self.data_time, self.losses, self.top1, self.top5],
            prefix=prefix,
        )

    def set_mode(self, mode):
        self.mode = mode
        if mode == "train":
            self.top1.summary_type = Summary.NONE
            self.top5.summary_type = Summary.NONE
        elif mode == "validate":
            self.top1.summary_type = Summary.AVERAGE
            self.top5.summary_type = Summary.AVERAGE

    def reset_metrics(self):
        self.batch_time.reset()
        self.data_time.reset()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()

    def update_batch(self, data_time, loss, batch_time, output, target):
        self.batch_history["mode"].append(self.mode)

        self.data_time.update(data_time)
        self.batch_history["data_time"].append(self.data_time.val)
        self.batch_history["data_time_avg"].append(self.data_time.avg)

        self.losses.update(loss, output.size(0))
        self.batch_history["loss"].append(self.losses.val)
        self.batch_history["loss_avg"].append(self.losses.avg)

        self.batch_time.update(batch_time)
        self.batch_history["batch_time"].append(self.batch_time.val)
        self.batch_history["batch_time_avg"].append(self.batch_time.val)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.top1.update(acc1[0], output.size(0))
        self.top5.update(acc5[0], output.size(0))
        self.batch_history["top1_accuracy"].append(self.top1.val)
        self.batch_history["top1_accuracy_avg"].append(self.top1.avg)
        self.batch_history["top5_accuracy"].append(self.top5.val)
        self.batch_history["top5_accuracy_avg"].append(self.top5.avg)

    def update_epoch(self):
        self.epoch_history["mode"].append(self.mode)
        self.epoch_history["data_time"].append(self.data_time.avg)
        self.epoch_history["loss"].append(self.losses.avg)
        self.epoch_history["batch_time"].append(self.batch_time.avg)
        self.epoch_history["top1_accuracy"].append(self.top1.avg)
        self.epoch_history["top5_accuracy"].append(self.top5.avg)
        