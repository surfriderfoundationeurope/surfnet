"""
Count detected objects in videos.
"""

import numpy as np
import os
import errno
import torch
import warnings
import datetime
import logging
import os.path as op
import argparse

from typing import Union

from plasticorigins.tracking.utils import (
    write_tracking_results_to_file,
    read_tracking_results,
)
from plasticorigins.tools.video_readers import IterableFrameReader
from plasticorigins.tracking.trackers import get_tracker
from plasticorigins.tracking.track_video import track_video
from plasticorigins.detection.yolo import load_model, predict_yolo
from plasticorigins.tracking.postprocess_and_count_tracks import filter_tracks
from plasticorigins.tracking.count import count_detected_objects


logger = logging.getLogger()


def check_file(filepath: str, raise_error: bool = True):
    """Check file, add log and raise error if file not found.

    Args:
        filepath (str): file path nto check.
        raise_error (bool, optional): If true raise error
            otherwise just log an error message. Defaults to True.

    Raises:
        FileNotFoundError: Couldn't find the provided path.
    """
    if not op.isfile(filepath):
        logger.error(f"ERROR : File {filepath} doesn't exists")
        if raise_error:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), filepath
            )


def main(args, display) -> tuple[int, int]:

    args.output_shape = tuple(int(s) for s in args.output_shape.split(","))

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    video_path = args.video_path
    check_file(video_path)

    if args.compare:
        check_file(args.video_count_path)

    device = torch.device(device)

    engine = get_tracker("EKF")

    logger.info("---Loading model...")
    model = load_model(
        args.weights, device, args.confidence_threshold, args.iou_threshold
    )
    logger.info("Model loaded.")

    def detector(frame):
        return predict_yolo(model, frame, size=args.size)

    transition_variance = np.load(
        os.path.join(args.noise_covariances_path, "transition_variance.npy")
    )
    observation_variance = np.load(
        os.path.join(args.noise_covariances_path, "observation_variance.npy")
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info(f"---Processing {video_path}")

    reader = IterableFrameReader(
        video_filename=video_path,
        skip_frames=args.skip_frames,
        output_shape=args.output_shape,
        progress_bar=not args.no_progress_bar,
        preload=args.preload_frames,
    )

    logger.info("Detecting...")

    detections = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        for frame in reader:
            detections.append(detector(frame))

    logger.info("Tracking...")

    results = track_video(
        reader,
        iter(detections),
        args,
        engine,
        transition_variance,
        observation_variance,
        display,
        is_yolo="yolo",
    )
    reader.video.release()

    # store unfiltered results
    datestr = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")

    output_filename = op.join(
        args.output_dir,
        op.splitext(os.path.basename(video_path))[0]
        + "_"
        + datestr
        + "_unfiltered.txt",
    )
    coord_mapping = reader.get_inv_mapping(args.downsampling_factor)
    write_tracking_results_to_file(
        results,
        coord_mapping,  # Scale the output back to original video size
        output_filename=output_filename,
    )

    logger.info("---Filtering...")
    results = read_tracking_results(output_filename)
    filtered_results = filter_tracks(results, args.kappa, args.tau)

    # store filtered results
    output_filename = op.join(
        args.output_dir,
        op.splitext(os.path.basename(video_path))[0]
        + "_"
        + datestr
        + "_filtered.txt",
    )

    write_tracking_results_to_file(
        filtered_results,
        lambda x, y: (x, y),  # No scaling, already scaled!
        output_filename=output_filename,
    )

    # Counting
    logger.info("--- Counting ...")

    n_det, n = count_detected_objects(
        filtered_results, args.video_count_path, args.compare
    )

    count_msg = f"{n_det} object(s) detected"
    if args.compare:
        count_msg = f"{count_msg} out of {n} ({round(n_det / n, 2)})"

    logger.info(count_msg)

    return n_det, n


def run(**kwargs) -> tuple[int, Union[int, None]]:
    """Count detected objects in a video.

    Example:
        import count_video_objects
        count_video_objects.run(weights='yolo.pt',
                                video_path='video/video1.mp4',
                                compare=True
                                video_count_path="video/video1.txt',
                                noise_covariances_path='data/tracking_parameters',
                                output_dir= '../runs/ct/'
        )

    Returns:
        tuple[int, Union[int, None]]: number of object detected and \
            the ground truth count if the compare parameter is true.

    """

    args = parse_opt(True)

    for k, v in kwargs.items():
        setattr(args, k, v)

    return main(args, display=None)


def parse_opt(known=False):

    parser = argparse.ArgumentParser(description="Counting objects in video")
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--noise_covariances_path", type=str)
    parser.add_argument(
        "--compare",
        action="store_true",
        default=False,
        help="Whether to compare to the ground truth value.",
    )
    parser.add_argument(
        "--video_count_path",
        type=str,
        help="Path to the file with the object count.",
        default=None,
    )
    parser.add_argument(
        "--output_dir", type=str, help="Filtering moving average window size"
    )
    parser.add_argument(
        "--tau",
        type=int,
        default=3,
        help="Minimum length of tracklet for filtering",
    )
    parser.add_argument("--kappa", type=int, default=5)
    parser.add_argument("--skip_frames", type=int, default=3)
    parser.add_argument("--confidence_threshold", type=float, default=0.35)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--downsampling_factor", type=int, default=1)
    parser.add_argument("--output_shape", type=str, default="960,544")
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--detection_batch_size", type=int, default=1)
    parser.add_argument("--preload_frames", action="store_true", default=False)
    parser.add_argument(
        "--no_progress_bar",
        action="store_true",
        default=False,
        help="Don't show progress bar.",
    )

    args = parser.parse_known_args()[0] if known else parser.parse_args()

    # validate some of the arguments
    if args.compare and args.video_count_path is None:
        logger.error(
            (
                "ERROR : With the compare option the path to the txt count "
                "file needs to be provided through the video_count_path "
                "argument"
            )
        )
        raise argparse.ArgumentError(
            args.video_count_path,
            "Missing video counts file " "while set to compare.",
        )

    return args


if __name__ == "__main__":

    args = parse_opt()
    display = None
    main(args, display)
