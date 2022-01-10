import json
import multiprocessing
import os
from typing import Dict, List, Tuple

import datetime
from flask import request, jsonify
from werkzeug.utils import secure_filename


# imports for tracking
import cv2
import numpy as np
import os
from detection.detect import detect
from tracking.postprocess_and_count_tracks import filter_tracks, postprocess_for_api
from tracking.utils import get_detections_for_video, write_tracking_results_to_file, read_tracking_results, gather_tracklets
from tracking.track_video import track_video
from tools.video_readers import IterableFrameReader
from tools.misc import load_model
from tracking.trackers import get_tracker
import torch


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


config_track = DotDict({
    "confidence_threshold": 0.5,
    "detection_threshold": 0.3,
    "downsampling_factor": 4,
    "noise_covariances_path": "data/tracking_parameters",
    "output_shape": (960,544),
    "skip_frames": 3, #3
    "arch": "mobilenet_v3_small",
    "device": "cpu",
    "detection_batch_size": 1,
    "display": 0,
    "kappa": 7, #7
    "tau": 4 #4
})


UPLOAD_FOLDER = '/tmp'  # folder used to store images or videos when sending files


def create_unique_folder(base_folder, filename):
    """Creates a unique folder based on the filename and timestamp
    """
    folder_name = os.path.splitext(os.path.basename(filename))[0] + "_out_"
    folder_name += datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    output_dir = os.path.join(base_folder, folder_name)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    return output_dir


def handle_post_request(upload_folder = UPLOAD_FOLDER):
    """main function to handle a post request.
    The file is in `request.files`

    Will create tmp folders for storing the file and intermediate results
    Outputs a json
    """
    print("---recieving request")
    if "file" in request.files:
        file = request.files['file']
    else:
        print("error no file in request")
        return None

    # file and folder handling
    filename = secure_filename(file.filename)
    print("---filename: "+filename)
    full_filepath = os.path.join(upload_folder, filename)
    output_dir = create_unique_folder(upload_folder, filename)
    if not os.path.isdir(upload_folder):
        os.mkdir(upload_folder)
    if os.path.isfile(full_filepath):
        os.remove(full_filepath)
    file.save(full_filepath)
    config_track.video_path = full_filepath
    config_track.output_dir = output_dir

    # launch the tracking
    filtered_results = track(config_track)

    # postprocess
    output_json = postprocess_for_api(filtered_results)
    response = jsonify(output_json)
    response.status_code = 200
    return response

def track(args):
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

    print(f'---Processing {args.video_path}')
    reader = IterableFrameReader(video_filename=args.video_path,
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
    display = None
    results = track_video(reader, iter(detections), args, engine, transition_variance, observation_variance, display)

    # store unfiltered results
    output_filename = os.path.splitext(args.video_path)[0] +'_unfiltered.txt'
    write_tracking_results_to_file(results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename)
    print('Filtering...')

    # read from the file
    results = read_tracking_results(output_filename)
    filtered_results = filter_tracks(results, config_track.kappa, config_track.tau)
    # store filtered results
    output_filename = os.path.splitext(args.video_path)[0] +'_filtered.txt'
    write_tracking_results_to_file(filtered_results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename)

    return filtered_results
