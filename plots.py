from math import log
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
from extension.losses import TrainLoss, TrainLossOneTerm
from torchvision.transforms.functional import center_crop
import pickle
import os
from base.centernet.models import create_model as create_model_centernet
from base.centernet.models import load_model as load_model_centernet
from common.utils import load_my_model, transform_test_CenterNet
from train_extension import spatial_transformer, mask_irrelevant_pixels
import cv2

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


def plot_single_image_and_heatmaps(image, heatmaps, normalize):

    _ , (ax0, ax1, ax2, ax3) = plt.subplots(1,4,figsize=(30,30))
    ax0.imshow(image)
    if not normalize: 
        kwargs = {'vmin':0, 'vmax':1, 'cmap':'gray'}
    else: 
        kwargs = {'cmap':'gray'}

    ax1.imshow(heatmaps[0].sigmoid_().cpu(),**kwargs)
    ax2.imshow(heatmaps[1].sigmoid_().cpu(),**kwargs)
    ax3.imshow(heatmaps[2].sigmoid_().cpu(),**kwargs)

    plt.show()
    plt.close()

def plot_heatmaps_and_gt(heatmaps, gt, normalize):

    _ , (ax0, ax1, ax2, ax3) = plt.subplots(1,4,figsize=(30,30))
    if not normalize: 
        kwargs = {'vmin':0, 'vmax':1, 'cmap':'gray'}
    else: 
        kwargs = {'cmap':'gray'}
    ax0.imshow(gt[0], **kwargs)
    ax1.imshow(heatmaps[0].sigmoid_().cpu(),**kwargs)
    ax2.imshow(heatmaps[1].sigmoid_().cpu(),**kwargs)
    ax3.imshow(heatmaps[2].sigmoid_().cpu(),**kwargs)
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

def plot_pickle_file(file_name):

    with open(file_name,'rb') as f :
        fig, axes= pickle.load(f)
    plt.show()
    plt.close()
    del fig
    del axes

def plot_pickle_folder(data_dir):
    pickle_files = [data_dir + file_name for file_name in sorted(os.listdir(data_dir)) if '.pickle' in file_name]
    for file_name in pickle_files:
        plot_pickle_file(file_name)

def plot_extracted_heatmaps(data_dir):
    pickle_files = [data_dir + file_name for file_name in sorted(os.listdir(data_dir)) if '.pickle' in file_name]
    for file_name in pickle_files:
        with open(file_name,'rb') as f :
            Z, Phi, center = pickle.load(f)
        print(center)
        plot_heatmaps_and_gt(Z, Phi, normalize=False)

def plot_base_heatmaps_centernet_official_repo(trained_model_weights_filename, images_folder, shuffle=True, fix_res=False, normalize=False):
    dataset = ImageFolder(images_folder, transform = lambda image: pre_process_centernet(image, fix_res), loader=cv2.imread)
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=1)
    model = create_model_centernet(arch='dla_34', heads={'hm':3,'wh':2,'reg':2}, head_conv=256)

    model = load_model_centernet(model, trained_model_weights_filename)
    for param in model.parameters():
        param.requires_grad = False
    model.to('cuda')
    print('Model loaded to GPU.')
    model.eval()
    with torch.no_grad():
        for image, _ in dataloader:
            image = image.to('cuda')
            predictions  = model(image)[-1]
            heatmaps = predictions['hm'][0]
            image = np.transpose(image.squeeze().cpu().numpy(), axes=[1, 2, 0])[...,::-1]
            image = image * (0.289, 0.274, 0.278) + (0.408, 0.447, 0.47)
            plot_single_image_and_heatmaps(image, heatmaps, normalize)

def plot_base_heatmaps_centernet_my_repo(trained_model_weights_filename, images_folder, shuffle=True, fix_res=False, normalize=False):
    dataset = ImageFolder(images_folder, transform = transform_test_CenterNet(fix_res))
    dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=1)
    model = create_model_centernet(arch='dla_34', heads={'hm':3,'wh':2}, head_conv=256)
    print(model)
    model = load_my_model(model, trained_model_weights_filename)
    for param in model.parameters():
        param.requires_grad = False
    model.to('cuda')
    print('Model loaded to GPU.')
    model.eval()
    with torch.no_grad():
        for image, _ in dataloader:
            image = image.to('cuda')
            predictions  = model(image)[-1]
            heatmaps = predictions['hm'][0]
            image = np.transpose(image.squeeze().cpu().numpy(), axes=[1, 2, 0])
            image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
            plot_single_image_and_heatmaps(image, heatmaps, normalize)

from common.utils import pre_process_centernet



def loss_experiments(model, dataloader, device='cuda'):
    model.train()
    loss = TrainLossOneTerm()
    test_shift_one_left = True
    with torch.no_grad():
        for i, (Z0, logit_Phi0, logit_Phi1, d_01) in enumerate(dataloader):

            Z0 = Z0.to(device)
            logit_Phi0 = logit_Phi0.to(device)
            # logit_Phi1 = logit_Phi1.to(device)
            # d_01 = d_01.to(device)
            if test_shift_one_left:
                h0 = torch.full_like(logit_Phi0, logit_Phi0.min()).to(device)
                max_position = np.unravel_index(torch.argmax(logit_Phi0).cpu().numpy(),logit_Phi0.shape)
                slided_max_position = np.array(max_position) + np.array((0,0,1,1))
                h0[slided_max_position[0],slided_max_position[1],slided_max_position[2],slided_max_position[3]] = logit_Phi0.max().item()
            
                # d_01 = torch.tensor([[-1,0]],dtype=torch.int32).to(device)
                # h0 = spatial_transformer(logit_Phi0, d_01)
                # h0 = mask_irrelevant_pixels(h0, d_01)
                loss(h0, logit_Phi0)


        # h0 = model(Z0)

        # h1 = spatial_transformer(h0, d_01)




if __name__ == '__main__':


    # images_folder = '/home/mathis/Documents/datasets/surfrider/other/test_synthetic_video_adour/'
    # centernet_trained_my_repo = 'external_pretrained_models/centernet_trained_my_repo.pth'
    # centernet_trained_official_repo = 'external_pretrained_models/centernet_trained_official_repo.pth'
    # plot_base_heatmaps_centernet_my_repo(centernet_trained_my_repo, images_folder, shuffle=False, fix_res=False, normalize=False)
    # plot_base_heatmaps_centernet_official_repo(centernet_trained_official_repo, images_folder, shuffle=False, fix_res=False, normalize=True)




            # else: 
            #     image = np.transpose(image.squeeze().cpu().numpy(), axes=[1, 2, 0]) 
            #     image = image * (0.229, 0.224, 0.225) +  (0.498, 0.470, 0.415)
    # plot_extracted_heatmaps('/home/mathis/Documents/datasets/surfrider/extracted_heatmaps/')

    # sftp_repo_dir = '/run/user/1000/gvfs/sftp:host=gpu1/home/infres/chagneux/repos/surfnet/'
    # plot_pickle_file(sftp_repo_dir+'verbose.pickle')
    # class Args(object):
    #     def __init__(self, focal, downsampling_factor):
    #         self.focal = focal
    #         self.downsampling_factor = downsampling_factor

    dataset_train =  SurfnetDataset('/home/mathis/Documents/datasets/surfrider/extracted_heatmaps/', split='train')
    loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)

    model = SurfNet(intermediate_layer_size=32)
    model.to('cuda')


    loss_experiments(model, loader_train)




    # args = Args(focal=True, downsampling_factor=4)

    # deeplab = load_deeplab_to_cuda('pretrained_models/model_83.pth')
    # # # surfnet = load_surfnet_to_cuda(32, 1, 'experiments/surfnet/focal_centernet_downsample_1_sigma2_2_alpha_2_beta_4_lr_0.0001/model_1.pth')
    # # from train_deeplab import get_dataset
    # # # dataset = SurfnetDataset(heatmaps_folder='/media/mathis/f88b9c68-1ae1-4ecc-a58e-529ad6808fd3/heatmaps_and_annotations/current/', split='train')
   
    # transform = lambda x: pre_process_centernet(x, scale=1, mean=)




    # # dataset, num_classes = get_dataset('/home/mathis/Documents/datasets/surfrider/images_subset/', 'surfrider','val', args)




    # # # loss = TrainLoss('focal_centernet',sigma2=2, alpha=2, beta=4)
    # # # model = get_model('deeplabv3__resnet50', 3, freeze_backbone=False, downsampling_factor=4)
    # # deeplab = load_deeplab_to_cuda('pretrained_models/deeplabv3__resnet101.pth')

    # # # test_model_output(deeplab, dataloader)
    # plot_deeplab_heatmaps(deeplab, dataloader)
    # # # plot_surfnet_pairs(surfnet, loss, dataloader)s
    # # test = next(iter(dataloader))

    # # plot_surfnet_heatmaps(deeplab, surfnet, dataloader)





