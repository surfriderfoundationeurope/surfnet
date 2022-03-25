import json
import multiprocessing
import os
from typing import Dict, List, Tuple

import datetime
from flask import request, jsonify
from werkzeug.utils import secure_filename
import logging

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
from tools.files import download_model_from_url, create_unique_folder
from tracking.trackers import get_tracker
import torch
# Optional Intel Deep Learning boost import
try:
    import intel_extension_for_pytorch as ipex
    print('Intel Deep Learning Boost module imported')
    intel_dl_boost = True
except ImportError:
    intel_dl_boost = False
    pass

from serving.config import id_categories, config_track


UPLOAD_FOLDER = '/tmp'  # folder used to store images or videos when sending files
logger = logging.getLogger()
if config_track.device is None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = config_track.device
device = torch.device(device)

engine = get_tracker('EKF')

logger.info('---Loading model...')
model = load_model(arch=config_track.arch, model_weights=config_track.model_weights, device=device)
if intel_dl_boost == True:
    model = ipex.optimize(model)
logger.info('---Model loaded.')


def handle_post_request(upload_folder = UPLOAD_FOLDER):
    """main function to handle a post request.
    The file is in `request.files`

    Will create tmp folders for storing the file and intermediate results
    Outputs a json
    """
    logger.info("---recieving request")
    if "file" in request.files:
        file = request.files['file']
    else:
        logger.error("error no file in request")

        return None

    # file and folder handling
    filename = secure_filename(file.filename)
    logger.info("---filename: "+filename)
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
    output_json = postprocess_for_api(filtered_results, id_categories)
    response = jsonify(output_json)
    response.status_code = 200
    return response

def track(args):

    detector = lambda frame: detect(frame, threshold=args.detection_threshold, model=model)

    transition_variance = np.load(os.path.join(args.noise_covariances_path, 'transition_variance.npy'))
    observation_variance = np.load(os.path.join(args.noise_covariances_path, 'observation_variance.npy'))

    logger.info(f'---Processing {args.video_path}')
    reader = IterableFrameReader(video_filename=args.video_path,
                                 skip_frames=args.skip_frames,
                                 output_shape=args.output_shape,
                                 progress_bar=True,
                                 preload=args.preload_frames)


    input_shape = reader.input_shape
    output_shape = reader.output_shape
    ratio_y = input_shape[0] / (output_shape[0] // args.downsampling_factor)
    ratio_x = input_shape[1] / (output_shape[1] // args.downsampling_factor)

    logger.info('---Detecting...')
    detections = get_detections_for_video(reader, detector, batch_size=args.detection_batch_size, device=device)

    logger.info('---Tracking...')
    display = None
    results = track_video(reader, iter(detections), args, engine, transition_variance, observation_variance, display)

    # store unfiltered results
    datestr = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    output_filename = os.path.splitext(args.video_path)[0] + "_" + datestr + '_unfiltered.txt'
    write_tracking_results_to_file(results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename)
    logger.info('---Filtering...')

    # read from the file
    results = read_tracking_results(output_filename)
    filtered_results = filter_tracks(results, config_track.kappa, config_track.tau)
    # store filtered results
    output_filename = os.path.splitext(args.video_path)[0] + "_" + datestr + '_filtered.txt'
    write_tracking_results_to_file(filtered_results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename)

    return filtered_results
