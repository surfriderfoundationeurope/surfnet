import logging
import os
from pathlib import Path

import numpy as np
import torch
from flask import jsonify, request
from werkzeug.utils import secure_filename

# imports for tracking
from plasticorigins.detection.detect import detect
from plasticorigins.tools.files import create_unique_folder
from plasticorigins.tools.misc import load_model
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
from plasticorigins.serving.config import config_track, id_categories

logger = logging.getLogger()
if config_track.device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = config_track.device
device = torch.device(device)

engine = get_tracker("EKF")

logger.info("---Loading model...")
model = load_model(
    arch=config_track.arch, model_weights=config_track.model_weights, device=device,
)
logger.info("---Model loaded.")


observation_variance = np.load(
    os.path.join(config_track.noise_covariances_path, "observation_variance.npy")
)
transition_variance = np.load(
    os.path.join(config_track.noise_covariances_path, "transition_variance.npy")
)


def handle_post_request():
    """main function to handle a post request.
    The file is in `request.files`

    Will create tmp folders for storing the file and intermediate results
    Outputs a json
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


def track(args):
    detector = lambda frame: detect(
        frame, threshold=args.detection_threshold, model=model
    )

    logger.info(f"---Processing {args.video_path}")
    reader = IterableFrameReader(
        video_filename=args.video_path,
        skip_frames=args.skip_frames,
        output_shape=args.output_shape,
        progress_bar=True,
        preload=args.preload_frames,
    )

    num_frames, fps = (
        int(reader.max_num_frames / (args.skip_frames + 1)),
        reader.fps,
    )

    input_shape = reader.input_shape
    output_shape = reader.output_shape
    ratio_y = input_shape[0] / (output_shape[0] // args.downsampling_factor)
    ratio_x = input_shape[1] / (output_shape[1] // args.downsampling_factor)

    logger.info("---Detecting...")
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
    )
    reader.video.release()
    # store unfiltered results
    output_filename = Path(args.output_dir) / "results_unfiltered.txt"
    write_tracking_results_to_file(
        results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename,
    )
    logger.info("---Filtering...")

    # read from the file
    results = read_tracking_results(output_filename)
    filtered_results = filter_tracks(results, args.kappa, args.tau)
    # store filtered results
    output_filename = Path(args.output_dir) / "results.txt"
    write_tracking_results_to_file(
        filtered_results,
        ratio_x=ratio_x,
        ratio_y=ratio_y,
        output_filename=output_filename,
    )

    return filtered_results, num_frames, fps
