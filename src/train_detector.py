import datetime
import os
import time

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from detection.coco_utils import get_surfrider
from detection import transforms
from detection.centernet.models import create_model
from detection.losses import FocalLoss
from detection import train_utils as utils


def get_dataset(dir_path, name, image_set, args):


    paths = {
        "surfrider": (dir_path, get_surfrider, args.num_classes),
    }
    p, ds_fn, num_classes = paths[name]

    train = image_set == 'train'
    transform = get_transform(train, num_classes, args)
    ds = ds_fn(p, image_set=image_set, transforms=transform)

    return ds, num_classes

def get_transform(train, num_classes, args):

    base_size = 540
    crop_size = (544, 960)
    if train:
        return transforms.TrainTransforms(base_size, crop_size, num_classes, args.downsampling_factor)
    return transforms.ValTransforms(base_size, crop_size, num_classes, args.downsampling_factor)

def evaluate(model, focal_loss, data_loader, device, num_classes):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    class_wise_loss = torch.zeros(size=(num_classes,))
    batch_nb = 0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            class_wise_loss += focal_loss(output, target)
            batch_nb += 1

    return class_wise_loss / batch_nb

def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    i = 0
    running_loss = 0.0
    for image, target in metric_logger.log_every(data_loader, print_freq, header):

        image, target = image.to(device), target.to(device)

        output = model(image)

        loss = criterion(output, target)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        writer.add_scalar('Training loss (mini-batch)',
                          loss.item(), epoch * len(data_loader) + i)
        writer.add_scalar('Learning rate (mini-batch)',
                          optimizer.param_groups[0]["lr"], epoch * len(data_loader) + i)
        i += 1

    lr_scheduler.step()
    writer.add_scalar('Training loss (epoch)', running_loss / (i+1), epoch)

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset, num_classes = get_dataset(
        args.data_path, args.dataset, "train", args)
    dataset_test, _ = get_dataset(args.data_path, args.dataset, "val", args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    model = create_model(arch=args.model, heads={'hm': num_classes}, head_conv=256)

    model.to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = torch.optim.Adam(model.parameters(), args.lr)


    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=0.1)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(
            checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    writer = SummaryWriter(args.logdir)

    start_time = time.time()


    criterion_train = FocalLoss(args.alpha, args.beta,
                            train=True)
    criterion_test = FocalLoss(args.alpha, args.beta,
                            train=False)


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, criterion_train, optimizer, data_loader,
                        lr_scheduler, device, epoch, args.print_freq, writer)

        class_wise_eval_focal_loss = evaluate(
            model, criterion_test, data_loader_test, device=device, num_classes=num_classes)
        writer.add_scalars(
            'Class-wise focal loss', {str(i): v for i, v in enumerate(class_wise_eval_focal_loss)}, epoch)
        print('Class-wise evaluation loss:',
                class_wise_eval_focal_loss.numpy())
        for name, param in model.named_parameters():
            writer.add_histogram(
                name, param.clone().cpu().data.numpy(), epoch)

        utils.save_on_master(
            {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args
            },
            os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='PyTorch Segmentation Training')

    parser.add_argument(
        '--data-path', default='/home/mathis/Documents/datasets/surfrider/images_subset/', help='dataset path')
    parser.add_argument('--dataset', default='surfrider', help='dataset name')
    parser.add_argument(
        '--model', default='deeplabv3__mobilenet_v3_large', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--num-classes', default=1, type=int,
                        help='number of classes (default: 1)')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=140, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr_step', default=140, type=int,
        help='when to decrease lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10,
                        type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--logdir', default='logs/deeplab')
    parser.add_argument('--output-dir', default='weights/deeplab')

    parser.add_argument('--downsampling-factor', default=4, type=int)
    parser.add_argument('--alpha', default=2, type=int)
    parser.add_argument('--beta', default=4, type=int)

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    args_dict = vars(args)

    with open(os.path.join(args.output_dir,'info.txt'),'w') as f:
        for k,v in args_dict.items():
            f.write(str(k)+':'+str(v)+'\n')
    main(args)
