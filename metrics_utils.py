from enum import Enum

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


class MetricsEngine():
    def __init__(self, use_accel, args=None):
        self.batch_time = AverageMeter("Time", use_accel, ":6.3f", Summary.NONE)
        self.data_time = AverageMeter("Data", use_accel, ":6.3f", Summary.NONE)
        self.losses = AverageMeter("Loss", use_accel, ":.4e", Summary.NONE)
        self.top1 = AverageMeter("Acc@1", use_accel, ":6.2f", Summary.NONE)
        self.top5 = AverageMeter("Acc@5", use_accel, ":6.2f", Summary.NONE)

    def configure_progress_meter(self, n_samples, epoch=None, mode=None):
        if mode == "train":
            prefix = "Training Epoch: [{}]".format(epoch)
        elif mode == "validate":
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
        if mode == 'train':
            self.top1.summary_type=Summary.NONE
            self.top5.summary_type=Summary.NONE
        elif mode == 'validate':
            self.top1.summary_type=Summary.AVERAGE
            self.top5.summary_type=Summary.AVERAGE

    def reset_metrics(self):
        self.batch_time.reset()
        self.data_time.reset()
        self.losses.reset()
        self.top1.reset()
        self.top5.reset()

    def calculate_task_spesific_metrics(self, model_predictions, targets):
        acc1, acc5 = accuracy(model_predictions, targets, topk=(1, 5))
        self.top1.update(acc1[0], model_predictions.size(0))
        self.top5.update(acc5[0], model_predictions.size(0))

    def enable_logging(self):
        raise NotImplemented
    
    def load_logging_checkpoint(self):
        raise NotImplemented