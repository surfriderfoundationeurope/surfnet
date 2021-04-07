from extension.datasets import SurfnetDatasetFlow
import torch 
from torch.utils.data import DataLoader
from extension.models import SurfNet
from torch.utils.tensorboard import SummaryWriter
from torch import sigmoid, logit
from torchvision.transforms.functional import center_crop
import matplotlib.pyplot as plt
from extension.losses import TrainLoss, TestLoss
from torchvision.transforms.functional import affine
from common.utils import warp_flow
# torch.autograd.set_detect_anomaly(True)

def spatial_transformer(heatmaps, displacement, device, dense_flow=True):

    if dense_flow: 
        return warp_flow(heatmaps, displacement, device)

    else: 
        DY = displacement[:,1] 
        DX = displacement[:,0]
        heatmaps =  torch.stack(tuple(affine(heatmap, angle = 0, translate = (dx,dy), shear = 0, scale=1) for (heatmap, dx, dy) in zip(heatmaps, DX, DY)))
        for j in range(len(displacement)):
            dx = DX[j]
            dy = DY[j]
            if dx > 0:
                heatmaps[j,:,:,:dx] = -50
            elif dx < 0:
                heatmaps[j,:,:,dx:] = -50

            if dy > 0:
                heatmaps[j,:,:dy,:] = -50
            elif dy < 0:
                heatmaps[j,:,dy:,:] = -50
        return heatmaps



def get_loaders(args):

    dataset_train = SurfnetDatasetFlow(args.data_path, split='train')
    dataset_test =  SurfnetDatasetFlow(args.data_path, split='test')

    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    print('Nb of batches train with batch_size {}:'.format(args.batch_size), len(dataset_train)// args.batch_size)
    print('Nb of batches test with batch_size 1:', len(dataset_test))

    return loader_train, loader_test

def train_one_epoch(model, criterion, optimizer, loader_train, lr_scheduler, device, epoch, writer):

    model.train()
    running_loss = 0.0
    verbose=False
    
    for i, (Z0, Phi0, Phi1, flow01) in enumerate(loader_train):

        Z0 = Z0.to(device)
        Phi0 = Phi0.to(device)
        Phi1 = Phi1.to(device)
        flow01 = flow01.to(device)

        h0 = model(Z0)

        h1 = spatial_transformer(h0, flow01, device)

        if verbose:
            Z1 = spatial_transformer(Z0, flow01, device)
            fig, ((ax0, ax1), (ax2, ax3),(ax4, ax5)) = plt.subplots(3,2, figsize=(10,10))

            ax0.imshow(sigmoid(Z0).detach().cpu()[0][0], cmap='gray', vmin=0, vmax=1)
            ax0.set_title('$\sigma(Z_0)$')

            ax1.imshow(sigmoid(Z1).detach().cpu()[0][0], cmap='gray', vmin=0, vmax=1)
            ax1.set_title('$\sigma(Z_1) = \sigma(T(Z_0, d_{01}))$')

            ax2.imshow(sigmoid(h0).detach().cpu()[0][0], cmap='gray', vmin=0, vmax=1)
            ax2.set_title('$\sigma(h_0)$')

            ax3.imshow(sigmoid(h1).detach().cpu()[0][0], cmap='gray', vmin=0, vmax=1)
            ax3.set_title('$\sigma(h_1) = \sigma(T(h_0, d_{01}))$')

            ax4.imshow(Phi0.cpu()[0][0], cmap='gray', vmin=0, vmax=1)
            ax4.set_title('$\Phi_0$')

            ax5.imshow(Phi1.cpu()[0][0], cmap='gray', vmin=0, vmax=1)
            ax5.set_title('$\Phi_1$')

            # plt.suptitle('$d_{01} = $'+str(d_01[0].detach().cpu().numpy()))
            plt.show()
            # with open('verbose.pickle','wb') as f:
            #     obj = (fig, ((ax2, ax3),(ax4, ax5)))
            #     pickle.dump(obj, f)
            plt.close()

        loss = criterion(h0, h1, Phi0, Phi1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('Training loss (mini-batch)', loss.item(), epoch * len(loader_train) + i)
        writer.add_scalar('Learning rate (mini-batch)', optimizer.param_groups[0]["lr"], epoch * len(loader_train) + i)

        running_loss+=loss.item()
        # if epoch == 0 and i == 0: 
        #     writer.add_graph(model, Z_0)
    lr_scheduler.step()
    writer.add_scalar('Training loss (epoch)', running_loss / (i+1), epoch)
   
def evaluate(model, criterion_test, loader_test, device, epoch, writer):

    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for batch_nb, (Z, Phi) in enumerate(loader_test):
            Z = Z.to(device)
            Phi = Phi.to(device)

            h = model(Z)

            loss = criterion_test(h, Phi)
            running_loss+=loss.item()

        writer.add_scalar('Validation loss (epoch)', running_loss/(batch_nb+1), epoch)

        # new_dir = 'eval_images/epoch_{}'.format(epoch)
        # os.mkdir(new_dir)
        # for i, (h_, phi_tilde) in enumerate(zip(h, Phi_tilde)):
        #     fig, (ax0, ax1) = plt.subplots(1,2)
        #     ax0.imshow(h_[0].cpu(), cmap='gray')
        #     ax1.imshow(phi_tilde[0].cpu(), cmap='gray')
        #     plt.savefig(new_dir + '/Eval_image_{}'.format(i))
        #     # plt.show()
        # plt.close()

def main(args):

    device = torch.device(args.device)
    # print(device)

    loader_train, loader_test = get_loaders(args)

    model = SurfNet(intermediate_layer_size=int(args.model.strip('surfnet')))
    model.to(device)

    params_to_optimize = [{"params": [p for p in model.parameters() if p.requires_grad]}]

    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    

    writer = SummaryWriter(args.log_dir)


    criterion_train = TrainLoss(args.alpha, args.beta)
    criterion_test = TestLoss(args.alpha, args.beta)

    model.conv3.bias.data.fill_(-2.19)

    for epoch in range(args.epochs):
        train_one_epoch(model, criterion_train, optimizer, loader_train, lr_scheduler, device, epoch, writer)
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        evaluate(model, criterion_test, loader_test, device, epoch, writer)
        torch.save(model.state_dict(), args.output_dir + '/model_{}.pth'.format(epoch))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Surfnet training')

    parser.add_argument('--data-path', default='/media/mathis/f88b9c68-1ae1-4ecc-a58e-529ad6808fd3/heatmaps_and_annotations/', help='dataset path')
    # parser.add_argument('--dataset', default='Heatma', help='dataset name')
    parser.add_argument('--model', default='surfnet32', help='model')
    # parser.add_argument('--aux-loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')


    # parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
    #                     help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('--log-dir', dest='log_dir', default='logs/surfnet')
    parser.add_argument('--output-dir', default='weights/surfnet')

    parser.add_argument('--loss', type=str)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--sigma2', type=float)

    parser.add_argument('--beta', type=float)
    parser.add_argument('--downsampling-factor', type=int)

    args = parser.parse_args()


    return args

if __name__ == '__main__':

    args = parse_args()
    main(args)



    # parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    # parser.add_argument('--output-dir', default='.', help='path where to save')
    # parser.add_argument('--resume', default='', help='resume from checkpoint')
    # parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    # parser.add_argument('--last-layer-only', dest='last_layer_only', default=True, action='store_true', help='Only retrain ultimate layer')
    # parser.add_argument('--freeze-backbone', dest='freeze_backbone', default=False, action='store_true', help='Freeze backbone weights')
    # parser.add_argument(
    #     "--test-only",
    #     dest="test_only",
    #     help="Only test the model",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "--pretrained",
    #     dest="pretrained",
    #     help="Use pre-trained models from the modelzoo",
    #     action="store_true",
    # )
    # # distributed training parameters
    # parser.add_argument('--world-size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
