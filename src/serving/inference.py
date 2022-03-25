import numpy as np
import json
import os
import os.path as op
from typing import Dict, List, Tuple

import datetime
from flask import request, jsonify
from werkzeug.utils import secure_filename
import logging
import warnings

# imports for tracking
from detection.yolo import load_model, predict_yolo
from tracking.postprocess_and_count_tracks import filter_tracks, postprocess_for_api
from tracking.utils import write_tracking_results_to_file, read_tracking_results
from tracking.track_video import track_video
from tools.video_readers import IterableFrameReader
from tools.files import download_model_from_url, create_unique_folder
from tracking.trackers import get_tracker

# config
from serving.config import id_categories, config_track

logger = logging.getLogger()

UPLOAD_FOLDER = '/tmp'  # folder used to store images or videos when sending files
logger.info('---Yolo model...')
# Yolo has warning problems, so we set an env variable to remove it
os.environ["VERBOSE"] = "False"
URL_MODEL = "https://github.com/surfriderfoundationeurope/IA_Pau/releases/download/v0.1/yolov5.pt"
FILE_MODEL = "yolov5.pt"
model_path = download_model_from_url(URL_MODEL, FILE_MODEL, logger)
model_yolo = load_model(model_path, config_track.device,
                                    config_track.yolo_conf_thrld,
                                    config_track.yolo_iou_thrld)


engine = get_tracker('EKF')

transition_variance = np.load(op.join(config_track.noise_covariances_path, 'transition_variance.npy'))
observation_variance = np.load(op.join(config_track.noise_covariances_path, 'observation_variance.npy'))


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
    full_filepath = op.join(upload_folder, filename)
    output_dir = create_unique_folder(upload_folder, filename)
    if not op.isdir(upload_folder):
        os.mkdir(upload_folder)
    if op.isfile(full_filepath):
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

    # Remove temp files (esp. the video):
    os.remove(full_filepath)

    return response


def track(args):
    detector = lambda frame: predict_yolo(model_yolo, frame, size=args.size, augment=False)

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

    detections = []
    logger.info('---Detecting...')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        for frame in reader:
            detections.append(detector(frame))

    logger.info('---Tracking...')
    display = None
    results = track_video(reader, iter(detections), args, engine, transition_variance, observation_variance, display, is_yolo=True)
    reader.video.release()
    # store unfiltered results
    datestr = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    output_filename = os.path.splitext(args.video_path)[0] + "_" + datestr + '_unfiltered.txt'
    write_tracking_results_to_file(results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename)
    logger.info('---Filtering...')

    # read from the file
    results = read_tracking_results(output_filename)
    filtered_results = filter_tracks(results, args.kappa, args.tau)
    # store filtered results
    output_filename = os.path.splitext(args.video_path)[0] + "_" + datestr + '_filtered.txt'
    write_tracking_results_to_file(filtered_results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename)

    return filtered_results
