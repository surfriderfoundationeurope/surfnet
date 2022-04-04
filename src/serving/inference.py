import json
import os
import os.path as op
from pathlib import Path
import datetime
from typing import Dict, List, Tuple

from werkzeug.utils import secure_filename
from flask import request, jsonify
import logging
import numpy as np
import warnings

# imports for tracking
from plasticorigins.detection.yolo import load_model, predict_yolo
from plasticorigins.tracking.postprocess_and_count_tracks import filter_tracks, postprocess_for_api
from plasticorigins.tracking.utils import write_tracking_results_to_file, read_tracking_results
from plasticorigins.tracking.track_video import track_video
from plasticorigins.tools.video_readers import IterableFrameReader
from plasticorigins.tools.files import download_from_url, create_unique_folder
from plasticorigins.tracking.trackers import get_tracker

# config
from serving.config import id_categories, config_track

logger = logging.getLogger()

logger.info('---Yolo model...')
# Yolo has warning problems, so we set an env variable to remove it
os.environ["VERBOSE"] = "False"
URL_MODEL = "https://github.com/surfriderfoundationeurope/IA_Pau/releases/download/v0.1/yolov5.pt"
FILE_MODEL = "yolov5.pt"
model_path = download_from_url(URL_MODEL, FILE_MODEL, "./models/", logger)
model_yolo = load_model(model_path, config_track.device,
                                    config_track.yolo_conf_thrld,
                                    config_track.yolo_iou_thrld)


engine = get_tracker('EKF')

transition_variance = np.load(op.join(config_track.noise_covariances_path, 'transition_variance.npy'))
observation_variance = np.load(op.join(config_track.noise_covariances_path, 'observation_variance.npy'))


def handle_post_request():
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
    working_dir = Path(create_unique_folder(config_track.upload_folder, filename))
    full_filepath = working_dir / filename
    if os.path.isfile(full_filepath):
        os.remove(full_filepath)
    file.save(full_filepath)
    config_track.video_path = full_filepath.as_posix()
    config_track.output_dir = working_dir.as_posix()

    # launch the tracking
    filtered_results, num_frames, fps = track(config_track)

    # postprocess
    output_json = postprocess_for_api(filtered_results, id_categories)
    output_json["fps"] = round(fps, 2)
    output_json["video_length"] = num_frames
    output_json["video_id"] = filename

    response = jsonify(output_json)
    response.status_code = 200

    # Remove temp files (esp. the video):
    os.remove(full_filepath)

    return response


def track(args):
    detector = lambda frame: predict_yolo(model_yolo, frame, size=args.size)

    logger.info(f'---Processing {args.video_path}')
    reader = IterableFrameReader(video_filename=args.video_path,
                                 skip_frames=args.skip_frames,
                                 output_shape=args.output_shape,
                                 progress_bar=True,
                                 preload=args.preload_frames)

    num_frames, fps = int(reader.max_num_frames / (args.skip_frames+1)), reader.fps
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
    results = track_video(reader, iter(detections), args, engine, transition_variance, observation_variance, None, is_yolo=True)
    reader.video.release()
    # store unfiltered results
    output_filename = Path(args.output_dir) / 'results_unfiltered.txt'
    write_tracking_results_to_file(results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename)
    logger.info('---Filtering...')

    # read from the file
    results = read_tracking_results(output_filename)
    filtered_results = filter_tracks(results, args.kappa, args.tau)
    # store filtered results
    output_filename = Path(args.output_dir) / 'results.txt'
    write_tracking_results_to_file(filtered_results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename)

    return filtered_results, num_frames, fps
