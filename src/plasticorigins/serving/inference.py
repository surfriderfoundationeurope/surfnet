"""The ``inference`` submodule provides several functions for tracking and handling post request.

The submodule configures :

- the device type
- the logger
- the prediction model

This submodule contains the following functions:

- ``handle_post_request()`` : Main function to handle a post request.
- ``track(args:argparse)`` : Tracking function for object detection in frame sequences.

"""

import json
import logging
import os
from pathlib import Path
import argparse
import warnings
import numpy as np
import torch
from flask import jsonify, request
from werkzeug.utils import secure_filename
from typing import List, Tuple

# imports for tracking
from plasticorigins.detection.detect import detect
from plasticorigins.tools.files import create_unique_folder, download_from_url

from plasticorigins.tools.video_readers import IterableFrameReader
from plasticorigins.tracking.postprocess_and_count_tracks import (
    filter_tracks,
    postprocess_for_api,
)
from plasticorigins.tracking.track_video import track_video
from plasticorigins.tracking.trackers import get_tracker
from plasticorigins.tracking.utils import (
    get_detections_for_video,
    read_tracking_results,
    write_tracking_results_to_file,
)
from plasticorigins.serving.config import id_categories

# centernet (deprecated) version
# from plasticorigins.serving.config import config_track

# yolo version
from plasticorigins.serving.config import config_track_yolo as config_track

logger = logging.getLogger()
if config_track.device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = config_track.device
device = torch.device(device)

engine = get_tracker("EKF")


if config_track.arch == "mobilenet_v3_small":
    from plasticorigins.tools.misc import load_model

    model = load_model(
        arch=config_track.arch,
        model_weights=config_track.model_weights,
        device=device,
    )
    logger.info("---Model mobilenet loaded.")
elif config_track.arch == "yolo":
    from plasticorigins.detection.yolo import load_model, predict_yolo

    model_path = download_from_url(
        config_track.url_model_yolo, config_track.file_model_yolo, "./models", logger
    )
    model_yolo = load_model(
        model_path,
        config_track.device,
        config_track.yolo_conf_thrld,
        config_track.yolo_iou_thrld,
    )
    logger.info("---Yolo model loaded.")
else:
    logger.error(f"unrecognized model {config_track.arch}")

observation_variance = np.load(
    os.path.join(config_track.noise_covariances_path, "observation_variance.npy")
)
transition_variance = np.load(
    os.path.join(config_track.noise_covariances_path, "transition_variance.npy")
)


def handle_post_request() -> json:

    """Main function to handle a post request. The file is in `request.files`. Will create temporary folders for storing the file and intermediate results. Outputs a json.

    Returns:
        A output json file.
    """

    logger.info("---receiving request")

    if "file" in request.files:
        file = request.files["file"]

    else:
        logger.error("error no file in request")
        return None

    # file and folder handling
    filename = secure_filename(file.filename)
    logger.info("--- received filename: " + filename)
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


def track(args: argparse) -> Tuple[List, int, int]:

    """Tracking function for object detection in frame sequences.

    Args:
        args (argparse): arguments for tracking process

    Returns:
        filtered_results (list): list of filtered tracks
        num_frames (int): max number of frames for tracking
        fps (int): number of frames per second (video speed)
    """

    if args.arch == "mobilenet_v3_small":
        detector = lambda frame: detect(
            frame, threshold=args.detection_threshold, model=model
        )
    elif args.arch == "yolo":
        detector = lambda frame: predict_yolo(
            model_yolo, frame, size=config_track.output_shape[0], augment=False
        )
    else:
        logger.error("bad model arch")

    logger.info(f"---Processing {args.video_path}")
    reader = IterableFrameReader(
        video_filename=args.video_path,
        skip_frames=args.skip_frames,
        output_shape=args.output_shape,
        progress_bar=True,
        preload=args.preload_frames,
        crop=args.crop,
    )

    num_frames, fps = (
        int(reader.max_num_frames / (args.skip_frames + 1)),
        reader.fps,
    )

    logger.info("---Detecting...")
    detections = []
    if args.arch == "yolo":
        # Catching warnings that come from a yolo bug:
        # "User provided device_type of \'cuda\', but CUDA is not available. Disabling"
        # should be fixed with more recent versions of yolo
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for frame in reader:
                detections.append(detector(frame))
    elif args.arch == "mobilenet_v3_small":
        detections = get_detections_for_video(
            reader, detector, batch_size=args.detection_batch_size, device=device
        )

    logger.info("---Tracking...")
    display = None

    results = track_video(
        reader,
        iter(detections),
        args,
        engine,
        transition_variance,
        observation_variance,
        display,
        is_yolo=args.arch == "yolo",
    )
    reader.video.release()
    # store unfiltered results
    output_filename = Path(args.output_dir) / "results_unfiltered.txt"
    coord_mapping = reader.get_inv_mapping(args.downsampling_factor)
    write_tracking_results_to_file(
        results,
        coord_mapping,  # Scale the output back to original video size
        output_filename=output_filename,
    )
    logger.info("---Filtering...")

    # read from the file
    results = read_tracking_results(output_filename)
    filtered_results = filter_tracks(results, args.kappa, args.tau)
    # store filtered results
    output_filename = Path(args.output_dir) / "results.txt"
    write_tracking_results_to_file(
        filtered_results,
        lambda x, y: (x, y),  # No scaling, already scaled!
        output_filename=output_filename,
    )

    return filtered_results, num_frames, fps
