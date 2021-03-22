from base.utils.presets import HeatmapExtractPreset
from torch.utils import data
from extension.models import SurfNet
from extension.datasets import SurfnetDataset
import torch 
from torch.utils.data import DataLoader
from torch import sigmoid
import matplotlib.pyplot as plt 
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import numpy as np
from extension.losses import TrainLoss
from torchvision.transforms.functional import center_crop
import pickle
import os
from train_base import get_transform
# from train_base import spatial_transformer


# def load_surfnet_to_cuda(intermediate_layer_size, downsampling_factor, checkpoint_name):
#     model = SurfNet(num_classes=1, intermediate_layer_size=intermediate_layer_size, downsampling_factor=downsampling_factor)
#     checkpoint = torch.load(checkpoint_name, map_location='cpu')
#     model.load_state_dict(checkpoint)
#     model.eval()
#     return model.to('cuda')

# def load_deeplab_to_cuda(model_name):
#     downsampling_factor=4
#     model = get_model('deeplabv3__mobilenet_v3_large', 3, freeze_backbone=False, downsampling_factor=downsampling_factor)
#     for param in model.parameters():
#         param.requires_grad = False
#     checkpoint = torch.load(model_name, map_location='cpu')
#     model.load_state_dict(checkpoint['model'])
#     return model.to('cuda')

def get_transform_images():

    transforms = []

    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)

def plot_deeplab_heatmaps(model, dataloader):
    model.eval()
    with torch.no_grad():
        for i, (image, _) in enumerate(dataloader): 
            image = image.to('cuda')
            predictions = model(image)['out'][0]
            fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,6, figsize=(10,10))
            image = np.transpose(image.squeeze().cpu().numpy(), axes=[1, 2, 0]) * (0.229, 0.224, 0.225) +  (0.498, 0.470, 0.415)
            ax0.imshow(image)
            ax0.set_title('Image')

            ax1.imshow(sigmoid(predictions[0]).cpu().numpy(),cmap='gray',vmin=0,vmax=1)
            ax1.set_title('Center heatmap class 0')

            ax2.imshow(sigmoid(predictions[1]).cpu().numpy(),cmap='gray',vmin=0,vmax=1)
            ax2.set_title('Center heatmap class 1')

            ax3.imshow(sigmoid(predictions[2]).cpu().numpy(),cmap='gray',vmin=0,vmax=1)
            ax3.set_title('Center heatmap class 2')

            ax4.imshow(predictions[3].cpu().numpy(),cmap='gray')
            ax4.set_title('Prediction height')

            ax5.imshow(predictions[4].cpu().numpy(),cmap='gray')
            ax5.set_title('Prediction width')

            plt.show()
            plt.close()

def plot_surfnet_heatmaps(model_deeplab, model_surfnet, dataloader):
    model_surfnet.eval()
    model_deeplab.eval()

    with torch.no_grad():
        for i, (image, _) in enumerate(dataloader): 
        # if i == 20: 
            image = image.to('cuda')
            predictions = model_deeplab(image)
            heatmap_deeplab = 1 - torch.nn.functional.softmax(predictions['out'][0], dim=0)[0]
            heatmap_surfnet = model_surfnet(heatmap_deeplab.unsqueeze(0).unsqueeze(0)).squeeze()
            fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(10,10))
            image = np.transpose(image.squeeze().cpu().numpy(), axes=[1, 2, 0]) * (0.229, 0.224, 0.225) +  (0.498, 0.470, 0.415)


            ax0.imshow(image)
            ax1.imshow(heatmap_deeplab.cpu().numpy(),cmap='gray',vmin=0,vmax=1)
            ax2.imshow(sigmoid(heatmap_surfnet).cpu().numpy(),cmap='gray',vmin=0,vmax=1)
            plt.savefig('result_{}'.format(i))
            # plt.show()
            plt.close()

def plot_surfnet_pairs(model_surfnet, loss, dataloader):

    with torch.no_grad():
        for i, (Z_0, Phi_0_tilde, Phi_1_tilde, d_01) in enumerate(dataloader):

            Z_0 = Z_0.to('cuda')
            Phi_0_tilde = Phi_0_tilde.to('cuda')
            Phi_1_tilde = Phi_1_tilde.to('cuda')
            d_01 = d_01.to('cuda')


            h_0 = model_surfnet(Z_0)

            h_1 = spatial_transformer(h_0, d_01)

            h, w = h_0.shape[2:]
            new_h, new_w = int(0.9*h), int(0.9*w)
            cropped_shape = (new_h, new_w)

            h_0 = center_crop(h_0, cropped_shape)
            h_1 = center_crop(h_1, cropped_shape)
            Phi_0_tilde = center_crop(Phi_0_tilde, cropped_shape)
            Phi_1_tilde = center_crop(Phi_1_tilde, cropped_shape)

            loss_value = loss(h_0, h_1, Phi_0_tilde, Phi_1_tilde)

            fig, ((ax2,ax3),(ax4, ax5)) = plt.subplots(2,2, figsize=(10,10))
            # ax0.imshow(center_crop(Z_0, cropped_shape).detach().cpu()[0][0],cmap='gray', vmin=0, vmax=1)
            # ax0.set_title('$Z_0$')

            # ax1.set_axis_off()

            ax2.imshow(sigmoid(h_0.cpu()[0][0]), cmap='gray', vmin=0, vmax=1)
            ax2.set_title('$\sigma(h_0)$')

            ax3.imshow(sigmoid(h_1.cpu()[0][0]), cmap='gray', vmin=0, vmax=1)
            ax3.set_title('$\sigma(h_1) = \sigma(T(h_0, d_{01}))$')

            ax4.imshow(sigmoid(Phi_0_tilde.cpu()[0][0]), cmap='gray', vmin=0, vmax=1)
            ax4.set_title('$\Phi_0$')

            ax5.imshow(sigmoid(Phi_1_tilde.cpu()[0][0]), cmap='gray', vmin=0, vmax=1)
            ax5.set_title('$\Phi_1$')

            plt.suptitle('Loss = {} '.format(loss_value)+'$d_{01} = $'+str(-d_01[0].cpu().numpy()))
            plt.savefig('result_{}'.format(i))
            plt.close()

def test_model_output(model, dataloader):
    model.to('cuda')
    model.eval()
    with torch.no_grad():
        # test = next(iter(dataloader))
        image, _ = next(iter(dataloader))
        image = image.to('cuda')

        output = model(image)
        print(output['out'].shape)
        print(model)

def plot_pickle(data_dir):
    pickle_files = [data_dir + file_name for file_name in sorted(os.listdir(data_dir)) if '.pickle' in file_name]
    for file_name in pickle_files:
        with open(file_name,'rb') as f :
            fig, axes= pickle.load(f)
        plt.show()
        plt.close()
        del fig
        del axes
def plot_pickle_file(file_name):

    with open(file_name,'rb') as f :
        fig, axes= pickle.load(f)
    plt.show()
    plt.close()
    del fig
    del axes

def plot_extracted_heatmaps(data_dir):
    pickle_files = [data_dir + file_name for file_name in sorted(os.listdir(data_dir)) if '.pickle' in file_name]
    for file_name in pickle_files:
        with open(file_name,'rb') as f :
            Z, Phi, center = pickle.load(f)
        fig, (ax0, ax1) = plt.subplots(1,2,figsize=(30,30))
        ax0.imshow(sigmoid(Z[0]),vmin=0, vmax=1, cmap='gray')
        ax0.set_title('$Z$')
        ax1.imshow(Phi, vmin=0, vmax=1, cmap='gray')
        ax1.set_title('$\Phi$')
        plt.suptitle('center = $'+str(center.detach().cpu().numpy()))
        plt.show()
        plt.close()


if __name__ == '__main__':

    plot_extracted_heatmaps('/home/infres/chagneux/datasets/surfrider_data/video_dataset/heatmaps_and_annotations/')

    # class Args(object):
    #     def __init__(self, focal, downsampling_factor):
    #         self.focal = focal
    #         self.downsampling_factor = downsampling_factor


    # args = Args(focal=True, downsampling_factor=4)

    # deeplab = load_deeplab_to_cuda('pretrained_models/model_83.pth')
    # # # surfnet = load_surfnet_to_cuda(32, 1, 'experiments/surfnet/focal_centernet_downsample_1_sigma2_2_alpha_2_beta_4_lr_0.0001/model_1.pth')
    # # from train_deeplab import get_dataset
    # # # dataset = SurfnetDataset(heatmaps_folder='/media/mathis/f88b9c68-1ae1-4ecc-a58e-529ad6808fd3/heatmaps_and_annotations/current/', split='train')
   
   
    # dataset = ImageFolder('/home/mathis/Documents/datasets/surfrider/other/Image_folder/', transform = get_transform_images())

    # # dataset, num_classes = get_dataset('/home/mathis/Documents/datasets/surfrider/images_subset/', 'surfrider','val', args)

    # dataloader = DataLoader(dataset, shuffle=True, batch_size=1)

    # # # loss = TrainLoss('focal_centernet',sigma2=2, alpha=2, beta=4)
    # # # model = get_model('deeplabv3__resnet50', 3, freeze_backbone=False, downsampling_factor=4)
    # # deeplab = load_deeplab_to_cuda('pretrained_models/deeplabv3__resnet101.pth')

    # # # test_model_output(deeplab, dataloader)
    # plot_deeplab_heatmaps(deeplab, dataloader)
    # # # plot_surfnet_pairs(surfnet, loss, dataloader)s
    # # test = next(iter(dataloader))

    # # plot_surfnet_heatmaps(deeplab, surfnet, dataloader)





