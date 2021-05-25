import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
from torchvision import datasets
from base.coco_utils import get_coco, get_surfrider, get_surfrider_old, get_surfrider_video_frames
from base.deeplab.models import get_model as get_model_deeplab
from base import presets
from torch.utils.tensorboard import SummaryWriter
from base.losses import cross_entropy
from base import train_utils as utils


def get_dataset(dir_path, name, image_set, args):
    def sbd(*args, **kwargs):
        return datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "voc": (dir_path, datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),
        "surfrider_old": (dir_path, get_surfrider_old, 4),
        "surfrider": (dir_path, get_surfrider, 1),
        "surfrider_video_frames": (dir_path, get_surfrider_video_frames, 1)
    }
    p, ds_fn, num_classes = paths[name]

    train = image_set == 'train'
    transform = get_transform(train, num_classes, args)
    ds = ds_fn(p, image_set=image_set, transforms=transform)

    return ds, num_classes

def get_transform(train, num_classes, args):
    base_size = 520
    crop_size = 512
    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        # Z = 1 - torch.nn.functional.softmax(output.squeeze(), dim=0)[0]
        # image = np.transpose(image.squeeze().cpu().numpy(), axes=[1, 2, 0]) * (0.229, 0.224, 0.225) +  (0.498, 0.470, 0.415)
        # writer.add_images('Test image', torch.from_numpy(image).unsqueeze(0), global_step=epoch)
        # writer.add_images('Test heatmap', Z.unsqueeze(0).unsqueeze(0), global_step=epoch)
        confmat.reduce_from_all_processes()

    return confmat

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

    model = get_model_deeplab(
        args.model, num_classes, freeze_backbone=args.freeze_backbone, downsampling_factor=args.downsampling_factor)


    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)


    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(
            checkpoint['model'], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    writer = SummaryWriter(args.logdir)

    if args.test_only:
        confmat = evaluate(model, data_loader_test,
                           device=device, num_classes=num_classes)
        print(confmat)
        return

    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, cross_entropy, optimizer, data_loader,
                        lr_scheduler, device, epoch, args.print_freq, writer)

        confmat = evaluate(
            model, data_loader_test, device=device, num_classes=num_classes)
        global_correct, average_row_correct, IoU, mean_IoU = confmat.get()
        writer.add_scalar('Global correct (epoch)', global_correct, epoch)
        writer.add_scalars('Average row correct (epoch)', {str(
            i): v for i, v in enumerate(average_row_correct)}, epoch)
        writer.add_scalars(
            'IoU (epoch)', {str(i): v for i, v in enumerate(IoU)}, epoch)
        writer.add_scalar('Mean IoU (epoch)', mean_IoU, epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(
                name, param.clone().cpu().data.numpy(), epoch)
        print(confmat)

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
    parser.add_argument('--aux-loss', action='store_true',
                        help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=140, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float,
                        help='initial learning rate')
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
    # parser.add_argument('--last-layer-only', dest='last_layer_only', default=True, action='store_true', help='Only retrain ultimate layer')
    parser.add_argument('--freeze-backbone', dest='freeze_backbone',
                        default=False, action='store_true', help='Freeze backbone weights')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    # parser.add_argument(
    #     "--pretrained",
    #     dest="pretrained",
    #     help="Use pre-trained models from the modelzoo",
    #     action="store_true",
    # )
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--logdir', default='logs/deeplab')
    parser.add_argument('--output-dir', default='weights/deeplab')

    parser.add_argument('--downsampling-factor', default=4, type=int)
    parser.add_argument('--alpha', default=2, type=int)
    parser.add_argument('--beta', default=4, type=int)
    parser.add_argument('--old_train', default=False, action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)