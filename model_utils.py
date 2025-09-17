import torch


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
