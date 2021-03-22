import torch
from extension.datasets import SingleVideoDataset
from base.utils.presets import HeatmapExtractPreset
import pickle 
import os 
from base.centernet.models import create_model, load_model
from common.utils import blob_for_bbox


def extract(args):
    downsampling_factor = args.downsampling_factor
    trained_model_weights_filename = args.weights

    video_folder = args.video_dir
    video_names = [video_name for video_name in sorted(os.listdir(video_folder)) if '.MP4' in video_name]

    transform = HeatmapExtractPreset()
    model = create_model(arch='dla34', dla={'hm':3,'wh':2,'reg':2}, head_conv=256)
    model = load_model(model, trained_model_weights_filename)
    print('Model loaded to GPU.')
    output_dir = args.output_dir 
    with torch.no_grad():
        for video_nb, video_name in enumerate(video_names): 
            print('Processing video {}'.format(video_nb))
            dataset = SingleVideoDataset(video_folder + video_name, transforms=transform)
            print('Video loaded')
            for frame_nb, (frame, annotation_dict) in enumerate(dataset):
                print('Extracting heatmap for frame {}'.format(frame_nb))
                frame = frame.to('cuda').unsqueeze(0)
                Z = model(frame)[0]|['hw'].cpu()
                Z = torch.sum(Z, axis=1)
                # h = surfnet(Z.unsqueeze(0).unsqueeze(0))
                Phi = torch.zeros(size=(Z.shape[1],Z.shape[2]))
                Phi, center = blob_for_bbox(annotation_dict['0']['bbox'], Phi, downsampling_factor)
                # print(center)
                data = (Z, Phi.unsqueeze(0), center)
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

    args = parser.parse_args()

    extract(args)
    
    




        



