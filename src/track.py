import numpy as np
import os
from detection.detect import detect
from tracking.utils import get_detections_for_video, write_tracking_results_to_file
from tools.video_readers import IterableFrameReader
from tools.misc import load_model
from tracking.trackers import get_tracker
import torch

from tracking.track_video import track_video, Display

def main(args, display):

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    device = torch.device(device)

    engine = get_tracker('EKF')

    print('---Loading model...')
    model = load_model(arch=args.arch, model_weights=args.model_weights, device=device)
    print('Model loaded.')

    detector = lambda frame: detect(frame, threshold=args.detection_threshold, model=model)

    transition_variance = np.load(os.path.join(args.noise_covariances_path, 'transition_variance.npy'))
    observation_variance = np.load(os.path.join(args.noise_covariances_path, 'observation_variance.npy'))

    video_filenames = [video_filename for video_filename in os.listdir(args.data_dir) if video_filename.endswith('.mp4')]

    for video_filename in video_filenames:
        print(f'---Processing {video_filename}')
        reader = IterableFrameReader(video_filename=os.path.join(args.data_dir, video_filename),
                                     skip_frames=args.skip_frames,
                                     output_shape=args.output_shape,
                                     progress_bar=True,
                                     preload=args.preload_frames)


        input_shape = reader.input_shape
        output_shape = reader.output_shape
        ratio_y = input_shape[0] / (output_shape[0] // args.downsampling_factor)
        ratio_x = input_shape[1] / (output_shape[1] // args.downsampling_factor)

        print('Detecting...')
        detections = get_detections_for_video(reader, detector, batch_size=args.detection_batch_size, device=device)

        print('Tracking...')
        results = track_video(reader, iter(detections), args, engine, transition_variance, observation_variance, display)

        output_filename = os.path.join(args.output_dir, video_filename.split('.')[0] +'.txt')
        write_tracking_results_to_file(results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Tracking')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--detection_threshold', type=float, default=0.3)
    parser.add_argument('--confidence_threshold', type=float, default=0.2)
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--downsampling_factor', type=int, default=1)
    parser.add_argument('--noise_covariances_path',type=str)
    parser.add_argument('--skip_frames',type=int,default=0)
    parser.add_argument('--output_shape',type=str,default='960,544')
    parser.add_argument('--arch', type=str, default='dla_34')
    parser.add_argument('--display', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--detection_batch_size',type=int,default=1)
    parser.add_argument('--preload_frames', action='store_true', default=False)
    args = parser.parse_args()

    display = None
    if args.display == 0:
        display = Display(on=False, interactive=True)
    elif args.display == 1:
        display = Display(on=True, interactive=True)
    elif args.display == 2:
        display = Display(on=True, interactive=False)

    args.output_shape = tuple(int(s) for s in args.output_shape.split(','))

    main(args, display)
