import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
from torchvision import datasets 

from base.utils.coco_utils import get_coco, get_surfrider, get_surfrider_focal
from base.deeplab.models import get_model as get_model_deeplab
from base.utils import presets
from torch.utils.tensorboard import SummaryWriter
from base.centernet.models import create_model as get_model_centernet
from base.losses import Loss
from base.utils import train_utils as utils

def get_dataset(dir_path, name, image_set, args):
    def sbd(*args, **kwargs):
        return datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "voc": (dir_path, datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21),
        "surfrider": (dir_path, get_surfrider, 4),
        "surfrider_focal": (dir_path, get_surfrider_focal, 1)
    }
    if args.focal:
        name = name+'_focal'
    p, ds_fn, num_classes = paths[name]

    train = image_set == 'train'
    transform = get_transform(train, num_classes, args)
    ds = ds_fn(p, image_set=image_set, transforms=transform)

    return ds, num_classes

def get_transform(train, num_classes, args):

    if args.focal:
        base_size = 520
        crop_size = 512
        downsampling_factor = args.downsampling_factor
        return  presets.SegmentationPresetTrainBboxes(base_size, crop_size, num_classes, downsampling_factor) if train else presets.SegmentationPresetEvalBboxes(crop_size, num_classes, downsampling_factor)
    else:
        base_size = 520
        crop_size = 480
        return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)


def cross_entropy(inputs, target):
    losses = {}
    losses['hm'] = nn.functional.cross_entropy(inputs['hm'], target, ignore_index=255)
    losses['aux'] = nn.functional.cross_entropy(inputs['aux'], target, ignore_index=255)
    if len(losses) == 1:
        return losses['hm']

    return losses['hm'] + 0.5 * losses['aux']


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

def evaluate_focal(model, focal_loss, data_loader, device, num_classes):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    class_wise_loss = torch.zeros(size=(num_classes,))
    batch_nb=0
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            class_wise_loss += focal_loss(output, target)
            batch_nb+=1

    return class_wise_loss / batch_nb



def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, writer):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    i=0
    running_loss = 0.0
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        print(image.shape)
        print(target.shape)
        
        image, target = image.to(device), target.to(device)

        output = model(image)

        # if i == 0:
        #     single_image = image[0].cpu().detach().numpy()
        #     single_target = target[0,:,:,:].cpu().detach()
        #     single_output_centers = torch.clamp(output['out'][0,:-2,:,:].cpu().detach().sigmoid_(), min=1e-4, max=1-1e-4)
        #     single_output_hw = output['out'][0,-2:,:,:].cpu().detach()
        #     single_image = np.transpose(single_image, axes=[1, 2, 0]) * (0.229, 0.224, 0.225) +  (0.498, 0.470, 0.415)

        #     fig, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9), (ax10, ax11, ax12, ax13, ax14)) = plt.subplots(3,5)

        #     ax0.imshow(single_image)
        #     ax1.set_axis_off()
        #     ax2.set_axis_off()
        #     ax3.set_axis_off()
        #     ax4.set_axis_off()

        #     ax5.imshow(single_target[0],cmap='gray', vmin=0, vmax=1)
        #     ax5.set_title('min_value:{}, max_value:{}'.format(single_target[0].min(), single_target[0].max()))

        #     ax6.imshow(single_target[1],cmap='gray', vmin=0, vmax=1)
        #     ax6.set_title('min_value:{}, max_value:{}'.format(single_target[1].min(), single_target[1].max()))

        #     ax7.imshow(single_target[2],cmap='gray', vmin=0, vmax=1)
        #     ax7.set_title('min_value:{}, max_value:{}'.format(single_target[2].min(), single_target[2].max()))

        #     ax8.imshow(single_target[3],cmap='gray', vmin=0, vmax=1)
        #     ax8.set_title('min_value:{}, max_value:{}'.format(single_target[3].min(), single_target[3].max()))

        #     ax9.imshow(single_target[4],cmap='gray', vmin=0, vmax=1)
        #     ax9.set_title('min_value:{}, max_value:{}'.format(single_target[4].min(), single_target[4].max()))

        #     ax10.imshow(single_output_centers[0],cmap='gray', vmin=0, vmax=1)
        #     ax10.set_title('min_value:{}, max_value:{},'.format(single_output_centers[0].min(), single_output_centers[0].max()))

        #     ax11.imshow(single_output_centers[1],cmap='gray', vmin=0, vmax=1)
        #     ax11.set_title('min_value:{},, max_value:{},'.format(single_output_centers[1].min(), single_output_centers[1].max()))

        #     ax12.imshow(single_output_centers[2],cmap='gray', vmin=0, vmax=1)
        #     ax12.set_title('min_value:{}, max_value:{}'.format(single_output_centers[2].min(), single_output_centers[2].max()))

        #     ax13.imshow(single_output_hw[0],cmap='gray')
        #     ax13.set_title('min_value:{}, max_value:{}'.format(single_output_hw[0].min(), single_output_hw[0].max()))

        #     ax14.imshow(single_output_hw[1],cmap='gray')
        #     ax14.set_title('min_value:{}, max_value:{}'.format(single_output_hw[1].min(), single_output_hw[1].max()))

            # with open('random_image_epoch_{}.pickle'.format(epoch),'wb') as f:
            #     data = (fig, ((ax0, ax1, ax2), (ax3, ax4, ax5), (ax6, ax7, ax8)))
            #     pickle.dump(data, f)
            # plt.close()
        # loss = 0.0

        # for i in range(len(output)):
        loss = criterion(output, target)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if lr_scheduler is not None: lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        
        writer.add_scalar('Training loss (mini-batch)', loss.item(), epoch * len(data_loader) + i)
        writer.add_scalar('Learning rate (mini-batch)', optimizer.param_groups[0]["lr"], epoch * len(data_loader) + i)
        i+=1

    lr_scheduler.step()
    writer.add_scalar('Training loss (epoch)', running_loss / (i+1), epoch)

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    dataset, num_classes = get_dataset(args.data_path, args.dataset, "train", args)
    dataset_test, _ = get_dataset(args.data_path, args.dataset, "val", args)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
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

    if args.focal:
        model = get_model_centernet(arch = args.model, heads = {'hm':num_classes,'wh':2}, head_conv=256)
        model.to(device)
    else:
        model = get_model_deeplab(args.model, num_classes, freeze_backbone=args.freeze_backbone, downsampling_factor=args.downsampling_factor)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # params_to_optimize = [
    #     {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
    #     {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    # ]
    # if args.aux_loss:
    #     params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
    #     params_to_optimize.append({"params": params, "lr": args.lr * 10})

    # optimizer = torch.optim.SGD(
    #     params_to_optimize,
    #     lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if args.focal: 
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    # lr_scheduler = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    writer = SummaryWriter(args.logdir)

    if args.test_only:
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
        print(confmat)
        return

    start_time = time.time()

    if args.focal:
        criterion_train = Loss(args.alpha, args.beta, train=True, centernet_output=True)
        criterion_test =  Loss(args.alpha, args.beta, train=False, centernet_output=True)
    else:
        criterion_train = cross_entropy

    # test = model.classifier[-1].bias.data

    if args.focal and args.model.split('__')[0] == 'deeplabv3':
        model.classifier[-1].bias.data[:-2].fill_(-2.19)
        model.aux_classifier[-1].bias.data.fill_(-2.19)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_one_epoch(model, criterion_train, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, writer)
        
        if not args.focal: 
            confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes)
            global_correct, average_row_correct, IoU, mean_IoU = confmat.get()
            writer.add_scalar('Global correct (epoch)', global_correct, epoch)
            writer.add_scalars('Average row correct (epoch)',{str(i):v for i,v in enumerate(average_row_correct)}, epoch)
            writer.add_scalars('IoU (epoch)',{str(i):v for i,v in enumerate(IoU)}, epoch)
            writer.add_scalar('Mean IoU (epoch)', mean_IoU, epoch)
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
            print(confmat)
        else: 
            class_wise_eval_focal_loss = evaluate_focal(model, criterion_test, data_loader_test, device=device, num_classes=num_classes)
            writer.add_scalars('Class-wise focal loss',{str(i):v for i,v in enumerate(class_wise_eval_focal_loss)}, epoch)
            print('Class-wise evaluation loss:', class_wise_eval_focal_loss.numpy())
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

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
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training')

    parser.add_argument('--data-path', default='/home/mathis/Documents/datasets/surfrider/images_subset/', help='dataset path')
    parser.add_argument('--dataset', default='surfrider', help='dataset name')
    parser.add_argument('--model', default='deeplabv3__mobilenet_v3_large', help='model')
    parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=140, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # parser.add_argument('--last-layer-only', dest='last_layer_only', default=True, action='store_true', help='Only retrain ultimate layer')
    parser.add_argument('--freeze-backbone', dest='freeze_backbone', default=False, action='store_true', help='Freeze backbone weights')
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
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--logdir', default='logs/deeplab')
    parser.add_argument('--output-dir', default='weights/deeplab')

    parser.add_argument('--downsampling-factor', default=4, type=int)
    parser.add_argument('--alpha', default=2, type=int)
    parser.add_argument('--beta', default=4, type=int)
    parser.add_argument('--focal', default=True, action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)