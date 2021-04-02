from base.deeplab.models import get_model
from extension.datasets import SingleVideoDataset
import pickle 
import os 
from base.centernet.models import create_model as create_model_centernet
from base.centernet.models import load_model as load_model_centernet
from common.utils import blob_for_bbox, pre_process_centernet, transform_test_CenterNet, transforms_test_deeplab
import numpy as np 
import matplotlib.pyplot as plt
from common.utils import load_my_model
import torch

def plot_single_image_heatmaps_and_gt(image, heatmaps, gt, normalize):

    _ , (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1,5,figsize=(30,30))
    ax0.imshow(image)
    if not normalize: 
        kwargs = {'vmin':0, 'vmax':1, 'cmap':'gray'}
    else: 
        kwargs = {'cmap':'gray'}

    ax1.imshow(heatmaps[0].sigmoid_(), **kwargs)
    ax2.imshow(heatmaps[1].sigmoid_(), **kwargs)
    ax3.imshow(heatmaps[2].sigmoid_(), **kwargs)
    ax4.imshow(gt, **kwargs)

    plt.show()

def extract_heatmaps_for_video_frames(model, args, transform, downsampling_factor):


    video_folder = args.input_dir
    video_names = [video_name for video_name in sorted(os.listdir(video_folder)) if '.MP4' in video_name]

    with torch.no_grad():
        for video_nb, video_name in enumerate(video_names): 
            print('Processing video {}'.format(video_nb))
            dataset = SingleVideoDataset(video_folder + video_name, transforms=transform)
            print('Video loaded')
            for frame_nb, (frame, annotation_dict) in enumerate(dataset):
                print('Extracting heatmap for frame {}'.format(frame_nb))
                frame = frame.to('cuda').unsqueeze(0)
                predictions = model(frame)[-1]
                Z = predictions['hm']

                Phi = np.zeros(shape=(Z.shape[-2],Z.shape[-1]))
                Phi, center = blob_for_bbox(annotation_dict['0']['bbox'], Phi, downsampling_factor)

                frame = np.transpose(frame.squeeze().cpu().numpy(), axes=[1, 2, 0])
                frame = frame * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
                # print(center)

                # plot_single_image_heatmaps_and_gt(frame, Z.cpu()[0], Phi, normalize=False)

                data = (Z.cpu().squeeze(), torch.from_numpy(Phi).unsqueeze(0), center)
                with open(args.output_dir + 'video_{:03d}_frame_{:03d}.pickle'.format(video_nb, frame_nb), 'wb') as f:
                    pickle.dump(data, f)


            del dataset

def extract(args):

    trained_model_weights_filename = args.weights

    if 'deeplab' in args.model:
        transform = transforms_test_deeplab
        model = get_model(args.model, 4, freeze_backbone=True, downsampling_factor=args.downsampling_factor, focal=False)
        model = load_my_model(model, trained_model_weights_filename)

    else:
        if args.my_repo: 
            transform = transform_test_CenterNet(fix_res=False)
            model = create_model_centernet(arch=args.model, heads={'hm':3,'wh':2}, head_conv=256)
            model = load_my_model(model, trained_model_weights_filename)

        else:
            transform = pre_process_centernet
            model = create_model_centernet(arch=args.model, heads={'hm':3,'wh':2,'reg':2}, head_conv=256)
            model = load_model_centernet(model, trained_model_weights_filename)

    for param in model.parameters():
        param.requires_grad = False
    model.to('cuda')
    print('Model loaded to GPU.')
    model.eval()

    if args.from_video:
        extract_heatmaps_for_video_frames(model, transform, args)
    
    else:
        extract_heatmaps_for_images_in_folder(model, transform, args)


if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser(description='Extracting heatmaps produced by base network')

    parser.add_argument('--input-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--weights')
    parser.add_argument('--downsampling-factor', type=int)
    parser.add_argument('--model')
    parser.add_argument('--my-repo',action='store_true')
    parser.add_argument('--from_videos',action='store_true')


    args = parser.parse_args()

    extract(args)
    
    




        



