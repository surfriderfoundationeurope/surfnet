from math import log
# from numba.np.ufunc import parallel
# from numba.np.ufunc.decorators import vectorize
from torchvision import datasets
from base.utils.presets import HeatmapExtractPreset
from torch.utils import data
from extension.models import SurfNet
from extension.datasets import SurfnetDataset
import torch 
from torch.utils.data import DataLoader, dataloader
from torch import sigmoid
import matplotlib.pyplot as plt 
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import numpy as np
from extension.losses import TestLoss, TrainLoss, TrainLossOneTerm
from torchvision.transforms.functional import center_crop
import pickle
import os
from base.centernet.models import create_model as create_model_centernet
from base.centernet.models import load_model as load_model_centernet
from common.utils import load_my_model, transform_test_CenterNet, nms
from train_extension import spatial_transformer, get_loaders
from train_base import get_dataset
import cv2
from common.utils import pre_process_centernet
from tqdm import tqdm
# from sklearn.metrics import roc_curve
from numba import jit


class Args(object):
    def __init__(self, focal, data_path, dataset, downsampling_factor, batch_size):
        self.focal = focal
        self.data_path = data_path
        self.dataset = dataset
        self.downsampling_factor = downsampling_factor
        self.batch_size = batch_size

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

def evaluate_extension_network_static_images(base_weights, extension_weights, data_path='data/surfrider_images/'):

    args = Args(focal=True,data_path=data_path,dataset='surfrider', downsampling_factor=4, batch_size=1)
    dataset_test, _ = get_dataset(args.data_path, 'surfrider', "val", args)
    dataloader_ = DataLoader(dataset_test, shuffle=True, batch_size=1)



    verbose = True
    enable_nms = True
    thres = 0.3 
    base_model = create_model_centernet('dla_34',heads={'hm':3,'wh':2}, head_conv=256)
    base_model = load_my_model(base_model, base_weights)
    extension_model = SurfNet(32)
    extension_model.load_state_dict(torch.load(extension_weights))
    for param in base_model.parameters():
        param.requires_grad = False
    for param in extension_model.parameters():
        param.requires_grad = False

    base_model.to('cuda')
    extension_model.to('cuda')
    base_model.eval()
    extension_model.eval()
    
    loss = TestLoss(alpha=2, beta=4)


    with torch.no_grad():
        running_loss_base = 0.0
        running_loss_extension = 0.0
        for batch_nb, (image, target) in enumerate(dataloader_):
            image = image.to('cuda')
            target = target.to('cuda')
            target = torch.max(target[:,:-2,:,:],dim=1,keepdim=True)[0]
            Z = base_model(image)[-1]['hm']
            Z = torch.max(Z,dim=1,keepdim=True)[0]
            h = extension_model(Z)

            loss_base = loss(Z,target)
            loss_extension = loss(h, target)
            running_loss_base+=loss_base
            running_loss_extension+=loss_extension


            Z = torch.sigmoid(Z)
            h = torch.sigmoid(h)
            if enable_nms:
                target = nms(target)
                Z = nms(Z)
                h = nms(h)
                if thres: 
                    Z[Z<thres] = 0
                    h[h<thres] = 0

            if verbose: 
                fig, (ax0, ax1, ax2, ax3) = plt.subplots(1,4, figsize=(20,20))
                image = np.transpose(image.squeeze().cpu().numpy(), axes=[1, 2, 0])
                image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
                ax0.imshow(image)
                ax0.set_title('Image')
                ax1.imshow(target[0][0].cpu(), cmap='gray',vmin=0, vmax=1)
                ax1.set_title('Ground truth')
                ax2.imshow(Z.cpu()[0][0],cmap='gray',vmin=0,vmax=1)
                ax2.set_title('Z, loss: {}'.format(loss_base))
                ax3.imshow(h.cpu()[0][0],cmap='gray',vmin=0,vmax=1)
                ax3.set_title('h, loss: {}'.format(loss_extension))
                plt.show()

    print('Evaluation loss base network:', running_loss_base.item()/(batch_nb+1))
    print('Evaluation loss extension network', running_loss_extension.item()/(batch_nb+1))

def evaluate_extension_network_video_frames(extension_weights, extracted_heatmaps_dir='data/extracted_heatmaps/'):

    verbose = False
    enable_nms = False
    thres = 0.3
    args = Args(focal=True, data_path=extracted_heatmaps_dir,dataset='surfrider',downsampling_factor=4, batch_size=1)
    loader_train, loader_test = get_loaders(args)
    extension_model = SurfNet(32)
    for param in extension_model.parameters():
        param.requires_grad = False
    extension_model.load_state_dict(torch.load(extension_weights))

    extension_model.to('cuda')
    extension_model.eval()

    loss = TestLoss(alpha=2, beta=4)

    with torch.no_grad():
        running_loss_base = 0.0
        running_loss_extension = 0.0
        for batch_nb, (Z, target) in tqdm(enumerate(loader_test)):
            Z = Z.to('cuda')
            target = target.to('cuda')
            h = extension_model(Z)

            loss_base = loss(Z, target)
            loss_extension = loss(h, target)

            running_loss_base+=loss_base
            running_loss_extension+=loss_extension

            Z = torch.sigmoid(Z)
            h = torch.sigmoid(h)
            if enable_nms:
                target = nms(target)
                Z = nms(Z)
                h = nms(h)
                if thres: 
                    Z[Z<thres] = 0
                    h[h<thres] = 0

            if verbose: 
                fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,20))
                ax1.imshow(target[0][0].cpu(), cmap='gray',vmin=0, vmax=1)
                ax1.set_title('Ground truth')
                ax2.imshow(Z.cpu()[0][0],cmap='gray',vmin=0,vmax=1)
                ax2.set_title('Z, loss: {}'.format(loss_base))
                ax3.imshow(h.cpu()[0][0],cmap='gray',vmin=0,vmax=1)
                ax3.set_title('h, loss: {}'.format(loss_extension))
                plt.show()

    print('Evaluation loss base network:', running_loss_base.item()/(batch_nb+1))
    print('Evaluation loss extension network', running_loss_extension.item()/(batch_nb+1))

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

def load_extension(extension_weights, intermediate_layer_size=32):
    extension_model = SurfNet(intermediate_layer_size)
    extension_model.load_state_dict(torch.load(extension_weights))
    for param in extension_model.parameters():
        param.requires_grad = False
    extension_model.to('cuda')
    extension_model.eval()
    return extension_model

def load_base(base_weights):
    base_model = create_model_centernet('dla_34', heads = {'hm':3,'wh':2}, head_conv=256)
    base_model = load_my_model(base_model, base_weights)
    for param in base_model.parameters():
        param.requires_grad = False 
    base_model.to('cuda')
    base_model.eval()
    return base_model

    return 

def extract_heatmaps_extension_from_base_heatmaps(extension_weights, input_dir):
    args = Args(focal=True, data_path=input_dir,dataset='surfrider', downsampling_factor=4, batch_size=1)
    _ , loader_test = get_loaders(args)
    extension_model  = load_extension(extension_weights)
    with torch.no_grad():
        base_predictions = list()
        extension_predictions = list()
        ground_truth = list() 

        for (Z, target) in tqdm(loader_test):
            Z = Z.to('cuda')
            h = extension_model(Z)
            base_predictions.append(torch.sigmoid(Z).cpu()[0][0])
            extension_predictions.append(torch.sigmoid(h).cpu()[0][0])
            ground_truth.append(target[0][0])

        with open('base_predictions.pickle','wb') as f: 
            pickle.dump(torch.stack(base_predictions),f)
        with open('extension_predictions.pickle','wb') as f: 
            pickle.dump(torch.stack(extension_predictions),f)     
        with open('ground_truth.pickle','wb') as f: 
            pickle.dump(torch.stack(ground_truth),f) 

def extract_heatmaps_extension_from_images(base_weights, extension_weights, input_dir):
    args = Args(focal=True, data_path=input_dir, dataset='surfrider',downsampling_factor=4, batch_size=1)
    dataset = get_dataset(input_dir,'surfrider','val', args)[0]
    loader_test = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)
    base_model = load_base(base_weights)
    extension_model = load_extension(extension_weights)

    with torch.no_grad():
        base_predictions = list()
        extension_predictions = list()
        ground_truth = list() 

        for (X, target) in tqdm(loader_test):
            X = X.to('cuda')
            target = torch.max(target[:,:-2,:,:], dim=1)[0]
            Z = torch.max(base_model(X)[-1]['hm'],dim=1,keepdim=True)[0]
            h = extension_model(Z)

            base_predictions.append(torch.sigmoid(Z).cpu()[0][0])
            extension_predictions.append(torch.sigmoid(h).cpu()[0][0])
            ground_truth.append(target[0])

        with open('base_predictions.pickle','wb') as f: 
            pickle.dump(torch.stack(base_predictions),f)
        with open('extension_predictions.pickle','wb') as f: 
            pickle.dump(torch.stack(extension_predictions),f)     
        with open('ground_truth.pickle','wb') as f: 
            pickle.dump(torch.stack(ground_truth),f)     

# @jit(nopython=False, parallel=True, fastmath=False)
def fast_ROC(gt, pred, thresholds):
    fpr, tpr = [], []

    num_samples = len(gt)
    P = gt.sum()
    N = num_samples - P 

    for thres in thresholds: 
        det = (pred > thres)
        TP_base = (det & gt).sum()
        FP_base = (det & ~gt).sum()

        fpr.append(FP_base/N)
        tpr.append(TP_base/P)

    return fpr, tpr

def compute_ROC_curves_brute(data_to_evaluate):
    with open(data_to_evaluate,'rb') as f: 
        all_data = pickle.load(f)
    # all_data = nms(all_data)
    predictions_base = all_data[:,0,:,:].numpy().flatten()
    predictions_extension = all_data[:,1,:,:].numpy().flatten()
    gt = all_data[:,2,:,:].eq(1).numpy().flatten()

    # fig, (ax0, ax1, ax2) = plt.subplots(3,1)
    # ax0.imshow(all_data[0,0,:,:], cmap='gray') #,vmin=0, vmax=1)
    # ax1.imshow(all_data[0,1,:,:], cmap='gray') #,vmin=0, vmax=1)
    # ax2.imshow(all_data[0,2,:,:], cmap='gray') #,vmin=0, vmax=1)
    # plt.show()

    thresholds = np.linspace(0.4,1,10)[::-1]

    # fpr_base, tpr_base, thresholds_base = roc_curve(gt,predictions_base) #sklearn verison 
    fpr_base, tpr_base = fast_ROC(gt, predictions_base, thresholds)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(fpr_base, tpr_base, thresholds, label='base')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_zlabel('Threshold')
    ax.set_title('ROC curve')
    plt.show()

@jit(nopython=True, fastmath=True, parallel=True)
def fast_prec_recall(gt, pred, thresholds):

    precision_list, recall_list = [], []

    for thres in thresholds: 
        detections = (pred >= thres)

        true_positives = 0 
        false_positives = 0 
        false_negatives = 0 

        for gt_frame, detection_frame in zip(gt, detections):
            # if thres > 0.2:
            #     _ , (ax0, ax1) = plt.subplots(2,1)
            #     ax0.scatter(np.arange(len(gt_frame)),gt_frame)
            #     ax0.set_title('Ground truth')
                
            #     ax1.scatter(np.arange(len(detection_frame)),detection_frame)
            #     ax1.set_title('Detections')
            #     plt.show()
            #     plt.close()

            positives_gt = gt_frame.sum()
            positives_pred = detection_frame.sum()

            if positives_pred > positives_gt: 
                true_positives+=positives_gt
                false_positives+=(positives_pred-positives_gt)
            else:
                true_positives+=positives_pred
                false_negatives+=(positives_gt-positives_pred)

        precision_list.append(true_positives/(true_positives+false_positives+1e-4) + 1e-4)
        recall_list.append(true_positives/(true_positives+false_negatives+1e-4) + 1e-4)

    return precision_list, recall_list

def plot_pr_curve(precision_list, recall_list, thresholds):
    f1 = 2*(precision_list*recall_list)/(precision_list+recall_list)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Recall', color=color)
    ax1.set_ylabel('Precision', color=color)
    ax1.plot(recall_list, precision_list, color=color)
    ax1.tick_params(axis='x', labelcolor=color)
    ax2 = ax1.twiny()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_xlabel('Threshold', color=color)  # we already handled the x-label with ax1
    ax2.plot(thresholds, f1, color=color)
    ax2.tick_params(axis='x', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    best_thres = thresholds[np.argmax(f1)]
    plt.suptitle('Optimum threshold {:.2f} at max f1 {:.2f}'.format(best_thres, max(f1)))
    return (fig,(ax1,ax2))

def pr_curve_from_file(filename,show=False):
    with open(filename, 'rb') as f: 
        fig, axes = plot_pr_curve(*pickle.load(f))
    fig.tight_layout()
    plt.savefig(filename.strip('.pickle'))
    if show:
        plt.show()
    plt.close()

def compute_precision_recall_nonlocal(gt, predictions, output_filename='evaluation', enable_nms=False, plot=False):

    if enable_nms: 
        predictions = nms(predictions)
    gt = gt.numpy()
    predictions = predictions.numpy()

    gt = gt.reshape(len(gt),-1)
    gt = (gt == 1)

    predictions = predictions.reshape(len(predictions),-1) 

    thresholds = np.linspace(0,1,1000)
    precision_list, recall_list = fast_prec_recall(gt, predictions, thresholds)
    precision_list, recall_list = np.array(precision_list), np.array(recall_list)
    with open(output_filename+'.pickle','wb') as f: 
        data = (precision_list, recall_list, thresholds)
        pickle.dump(data,f)
    if plot: 
        plot_pr_curve(precision_list, recall_list, thresholds)

if __name__ == '__main__':

    # extension_weights = 'experiments/extension/surfnet32_alpha_2_beta_4_lr_1e-5_lr_reduced_epoch_15_multi_obj_no_obj/model_49.pth'
    # input_dir = 'data/extracted_heatmaps/'
    # extract_heatmaps_extension_from_base_heatmaps(extension_weights=extension_weights, input_dir=input_dir)
    # extract_heatmaps_extension_from_images(base_weights='external_pretrained_models/centernet_pretrained.pth', extension_weights='external_pretrained_models/surfnet32.pth', input_dir='data/surfrider_images')
   
    # compute_ROC_curves_brute('data_to_evaluate.pickle')

    eval_dir = 'experiments/evaluations/multi_object_synthetic_videos_trained_including_no_obj/'
    with open(eval_dir+'ground_truth.pickle','rb') as f: 
        gt = pickle.load(f)
    with open(eval_dir+'extension_predictions.pickle','rb') as f: 
        predictions_extension = pickle.load(f)
    with open(eval_dir+'base_predictions.pickle','rb') as f: 
        predictions_base = pickle.load(f)
    # permutation = np.random.permutation(gt.shape[0])

    # gt = gt[permutation]
    # predictions_base = predictions_base[permutation]
    # predictions_extension = predictions_extension[permutation]
    
    compute_precision_recall_nonlocal(gt, predictions_base, output_filename='Evaluation base')
    compute_precision_recall_nonlocal(gt, predictions_base, output_filename='Evaluation base nms', enable_nms=True)
    compute_precision_recall_nonlocal(gt, predictions_extension, output_filename='Evaluation extension')
    compute_precision_recall_nonlocal(gt, predictions_extension, output_filename='Evaluation extension nms', enable_nms=True)

    


    # # extract_heatmaps_extension('external_pretrained_models/surfnet32.pth','data/extracted_heatmaps/')

    # for gt_, prediction_base, predictions_extension in (zip(gt,predictions_base, predictions_extension)):
    #     fig, (ax0, ax1, ax2) = plt.subplots(1,3)
    #     ax0.imshow(gt_,cmap='gray',vmin=0,vmax=1)
    #     ax0.set_title('Ground truth')
    #     ax1.imshow(prediction_base,cmap='gray',vmin=0,vmax=1)
    #     ax1.set_title('Base')
    #     ax2.imshow(predictions_extension,cmap='gray',vmin=0,vmax=1)
    #     ax2.set_title('Extension')
    #     plt.show()
    #     plt.close()
    
    pr_curve_from_file('Evaluation base.pickle')
    pr_curve_from_file('Evaluation base nms.pickle')
    pr_curve_from_file('Evaluation extension.pickle')
    pr_curve_from_file('Evaluation extension nms.pickle')




























































    # loaded_data = torch.tensor(load_data_for_eval('experiments/evaluations/temp_heatmaps/'))
    # test = 0 

    # extract_heatmaps_extension(extension_weights='external_pretrained_models/surfnet32.pth',input_dir='data/extracted_heatmaps/')
    # load_heatmaps('experiments/evaluations/temp_heatmaps/')
    # evaluate_extension_network_static_images(base_weights='external_pretrained_models/centernet_pretrained.pth', extension_weights='external_pretrained_models/surfnet32.pth')

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

    # dataset_test =  SurfnetDataset('/home/mathis/Documents/datasets/surfrider/extracted_heatmaps/', split='train')
    # loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)

    # model = SurfNet(intermediate_layer_size=32)
    # model.to('cuda')


    # loss_experiments(model, loader_train)




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





