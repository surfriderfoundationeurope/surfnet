from base.deeplab.models import get_model
import torch
from extension.datasets import SingleVideoDataset
from base.utils.presets import HeatmapExtractPreset
import pickle 
import os 
from base.centernet.models import create_model as create_model_centernet
from base.centernet.models import load_model as load_model_centernet
from common.utils import blob_for_bbox, pre_process_centernet
import numpy as np 
import matplotlib.pyplot as plt


def load_deeplab_to_cuda(model_name, model_weights=None):
    return 

def extract(args):
    downsampling_factor = args.downsampling_factor
    trained_model_weights_filename = args.weights

    video_folder = args.video_dir
    video_names = [video_name for video_name in sorted(os.listdir(video_folder)) if '.MP4' in video_name]

    if 'deeplab' in args.model:
        transform = HeatmapExtractPreset()
        model = get_model(args.model, 4, freeze_backbone=True, downsampling_factor=args.downsampling_factor,focal=False)
        model = load_deeplab_to_cuda(args.model_name)
    else:
        transform = pre_process_centernet
        model = create_model_centernet(arch=args.model, heads={'hm':3,'wh':2,'reg':2}, head_conv=256)
        model = load_model_centernet(model, trained_model_weights_filename)
    for param in model.parameters():
        param.requires_grad = False
    model.to('cuda')
    print('Model loaded to GPU.')
    model.eval()

    output_dir = args.output_dir 
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
                # fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1,5,figsize=(20,20))
                # frame = np.transpose(frame.squeeze().cpu().numpy(), axes=[1, 2, 0])[...,::-1]
                # frame = frame * (0.289, 0.274, 0.278) + (0.408, 0.447, 0.47)
                # ax0.imshow(frame)
                # ax1.imshow(Phi, vmin=0, vmax=1, cmap='gray')
                # ax2.imshow(torch.sigmoid(Z[0][0]).cpu(), vmin=0, vmax=1, cmap='gray')
                # ax3.imshow(torch.sigmoid(Z[0][1]).cpu(), vmin=0, vmax=1, cmap='gray')
                # ax4.imshow(torch.sigmoid(Z[0][2]).cpu(), vmin=0, vmax=1, cmap='gray')
                # plt.show()
                # plt.close()
                data = (Z, torch.from_numpy(Phi).unsqueeze(0), center)
                with open(output_dir + 'video_{:03d}_frame_{:03d}.pickle'.format(video_nb, frame_nb), 'wb') as f:
                    pickle.dump(data, f)
            del dataset

if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser(description='Extracting heatmaps from trained Deeplab')

    parser.add_argument('--video-dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--weights')
    parser.add_argument('--downsampling-factor', type=int)
    parser.add_argument('--model')

    args = parser.parse_args()

    extract(args)
    
    




        



