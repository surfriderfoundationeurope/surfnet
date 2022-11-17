"""The ``track_video`` submodule provides several functions to track trashs on videos.

This submodule contains the following functions :

- associate_detections_to_trackers(detections_for_frame:List[ndarray], confs:array[Any,dtype[float64]], labels:array[Any,dtype[int64]],
    trackers:Dict[int,Tracker], flow01:ndarray, confidence_threshold:float) : Associate detections to trackers.
- build_confidence_function_for_trackers(trackers:Dict[int,Tracker], flow01:ndarray) : Build confidence function for trackers.
- init_trackers(engine:Any, detections:List[ndarray], confs:array[Any,dtype[float64]], labels:array[Any,dtype[int64]], frame_nb:int, state_variance:ndarray[Any,dtype[float64]], observation_variance:ndarray[Any,dtype[float64]],
    delta:float): Initializes the trackers based on detections.
- interpret_detection(detections_for_frame:List[ndarray], downsampling_factor:Union[int,float], is_yolo:bool=False) : Normalizes the detections depending whether they come from Centernet or Yolo model
- track_video(reader:Any, detections:List[ndarray], args:argparse, engine:Any, transition_variance:ndarray[Any,dtype[float64]],
    observation_variance:ndarray[Any,dtype[float64]], display:Any, is_yolo:bool=False) : Original version of tracking trashs on video.

"""

import argparse
import numpy as np
from numpy import array, ndarray, dtype, float64
from typing import Any, Dict, Iterable, List, Tuple, Union

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean

from plasticorigins.tools.optical_flow import compute_flow
from plasticorigins.tracking.utils import in_frame
from plasticorigins.tracking.trackers import Tracker


def init_trackers(
    engine: Any,
    detections: List[ndarray],
    confs: array,
    labels: array,
    frame_nb: int,
    state_variance: ndarray[Any, dtype[float64]],
    observation_variance: ndarray[Any, dtype[float64]],
    delta: float,
) -> List[Tracker]:

    """Initializes the trackers based on detections.

    Args:
        engine (Any): the engine tracker
        detections (List[ndarray]): list of detected object positions with center coordinates such as ``list[np.array([[xcenter, ycenter], ...]), ...]``
        confs (array[Any,dtype[float64]]): array of the confidence values
        labels (array[Any,dtype[int64]]): array of the object classes
        frame_nb (int): number of frames
        state_variance (ndarray[Any,dtype[float64]]): state variances
        observation_variance (ndarray[Any,dtype[float64]]): observation variances
        delta (float): threshold in probabilty between [0,1] (confidence threshold). Probabilty of a point to belong to a specific neighbourhood.

    Returns:
        trackers (List[Tracker]): the trackers based on detections and previous parameters
    """

    trackers = []

    for detection, conf, label in zip(detections, confs, labels):
        tracker_for_detection = engine(
            frame_nb,
            detection,
            conf,
            label,
            state_variance,
            observation_variance,
            delta,
        )
        trackers.append(tracker_for_detection)

    return trackers


def build_confidence_function_for_trackers(
    trackers: Dict[int, Tracker], flow01: ndarray
) -> Tuple[List[int], List]:

    """Build confidence function for trackers.

    Args:
        trackers (Dict[int,Tracker]): the trackers
        flow01 (ndarray): the reference flow

    Returns:
        tracker_nbs (List[int]): the list of tracker numbers
        confidence_functions (List[Any]): the list of confidence functions
    """

    tracker_nbs = []
    confidence_functions = []

    for tracker_nb, tracker in enumerate(trackers):
        if tracker.enabled:
            tracker_nbs.append(tracker_nb)
            confidence_functions.append(tracker.build_confidence_function(flow01))

    return tracker_nbs, confidence_functions


def associate_detections_to_trackers(
    detections_for_frame: List[ndarray],
    confs: array,
    labels: array,
    trackers: Dict[int, Tracker],
    flow01: ndarray,
    confidence_threshold: float,
) -> List:

    """Associate detections to trackers.

    Args:
        detections_for_frame (List[np.ndarray]): list of detected object positions with center coordinates such as ``list[np.array([[xcenter, ycenter], ...]), ...]``
        confs (array[float]): array of the confidence values
        labels (array[int]): array of the object labels
        trackers (Dict[int,Tracker]): the trackers
        flow01 (array): the reference flow
        confidence_threshold (float): confidence threshold for detection in [0,1]

    Returns:
        assigned_trackers (List[int]): the list of assigned trackers
    """

    tracker_nbs, confidence_functions = build_confidence_function_for_trackers(
        trackers, flow01
    )
    assigned_trackers = [None] * len(detections_for_frame)

    if len(tracker_nbs):
        cost_matrix = np.zeros(shape=(len(detections_for_frame), len(tracker_nbs)))

        for detection_nb, (detection, conf, label) in enumerate(
            zip(detections_for_frame, confs, labels)
        ):

            for tracker_id, confidence_function in enumerate(confidence_functions):
                score = confidence_function(detection)
                cls_score = trackers[tracker_id].cls_score_function(conf, label)

                if cls_score < 0.5:
                    score = score * 0.1  # if wrong class, reduce the score, to tweak

                if score > confidence_threshold:
                    cost_matrix[detection_nb, tracker_id] = score

                else:
                    cost_matrix[detection_nb, tracker_id] = 0

        row_inds, col_inds = linear_sum_assignment(cost_matrix, maximize=True)

        for row_ind, col_ind in zip(row_inds, col_inds):
            if cost_matrix[row_ind, col_ind] > confidence_threshold:
                assigned_trackers[row_ind] = tracker_nbs[col_ind]

    return assigned_trackers


def interpret_detection(
    detections_for_frame: List[ndarray],
    downsampling_factor: Union[int, float],
    is_yolo: bool = False,
) -> Tuple[List[ndarray], array, array]:

    """Normalizes the detections depending whether they come from Centernet or Yolo model

    Args:
        detections_for_frame (List[ndarray]): list of detected object positions with center coordinates such as ``list[np.array([[xcenter, ycenter], ...]), ...]``
        downsampling_factor (Union[int,float]): downsampling factor
        is_yolo (bool): ``True`` if the model used for detection is Yolo. ``False`` if the model used for detection is Centernet.

    Returns:
        detections_for_frame (List[ndarray]): list of detected object positions with center coordinates such as ``list[np.array([[xcenter, ycenter], ...]), ...]``
        confs (array[Any,dtype[float64]]): array of the confidence values after normalization
        labels (array[Any,dtype[int64]]): array of the object labels after normalization
    """

    if not is_yolo:
        confs = [1.0] * len(detections_for_frame)
        labels = [0] * len(detections_for_frame)
        return detections_for_frame, confs, labels

    else:
        detections_for_frame, confs, labels = detections_for_frame
        # get center
        detections_for_frame = detections_for_frame[..., 0:2] / downsampling_factor
        return detections_for_frame, confs, labels


def track_video(
    reader: Any,
    detections: List[ndarray],
    args: argparse,
    engine: Any,
    transition_variance: ndarray[Any, dtype[float64]],
    observation_variance: ndarray[Any, dtype[float64]],
    display: Any,
    is_yolo: bool = False,
) -> Iterable:

    """Original version of tracking trashs on video. Expects detections in the format ``list[np.array([[xcenter, ycenter], ...]), ...]``.

    Args:
        reader (Any): the video reader
        detections (List[ndarray]): array of detected object positions
        args (argparse): array of the confidence values
        engine (Any): the engine tracker
        transition_variance (ndarray[Any,dtype[float64]]): transition variances
        observation_variance (ndarray[Any,dtype[float64]]): observation variances
        display (Any): to display the tracker
        is_yolo (bool): ``True`` if the model used for detection is Yolo. ``False`` if the model used for detection is Centernet.

    Returns:
        results (Iterable): the tracker based on detections and previous parameters
    """

    init = False
    trackers = dict()
    frame_nb = 0
    frame0 = next(reader)
    detections_for_frame = next(detections)
    detections_for_frame, confs, labels = interpret_detection(
        detections_for_frame, args.downsampling_factor, is_yolo
    )

    max_distance = euclidean(reader.output_shape, np.array([0, 0]))
    delta = 0.005 * max_distance

    if display is not None and display.on:
        display.display_shape = (
            reader.output_shape[0] // args.downsampling_factor,
            reader.output_shape[1] // args.downsampling_factor,
        )
        display.update_detections_and_frame(detections_for_frame, frame0)

    if len(detections_for_frame):
        trackers = init_trackers(
            engine,
            detections_for_frame,
            confs,
            labels,
            frame_nb,
            transition_variance,
            observation_variance,
            delta,
        )
        init = True

    if display is not None and display.on:
        display.display(trackers)

    for frame_nb, (frame1, detections_for_frame) in enumerate(
        zip(reader, detections), start=1
    ):
        detections_for_frame, confs, labels = interpret_detection(
            detections_for_frame, args.downsampling_factor, is_yolo
        )

        if display is not None and display.on:
            display.update_detections_and_frame(detections_for_frame, frame1)

        if not init:
            if len(detections_for_frame):
                trackers = init_trackers(
                    engine,
                    detections_for_frame,
                    confs,
                    labels,
                    frame_nb,
                    transition_variance,
                    observation_variance,
                    delta,
                )
                init = True
        else:
            new_trackers = []
            flow01 = compute_flow(frame0, frame1, args.downsampling_factor)

            if len(detections_for_frame):
                assigned_trackers = associate_detections_to_trackers(
                    detections_for_frame,
                    confs,
                    labels,
                    trackers,
                    flow01,
                    args.confidence_threshold,
                )

                for detection, conf, label, assigned_tracker in zip(
                    detections_for_frame, confs, labels, assigned_trackers
                ):
                    if in_frame(detection, flow01.shape[:-1]):
                        if assigned_tracker is None:
                            new_trackers.append(
                                engine(
                                    frame_nb,
                                    detection,
                                    conf,
                                    label,
                                    transition_variance,
                                    observation_variance,
                                    delta,
                                )
                            )
                        else:
                            trackers[assigned_tracker].update(
                                detection, conf, label, flow01, frame_nb
                            )

            for tracker in trackers:
                tracker.update_status(flow01)

            if len(new_trackers):
                trackers.extend(new_trackers)

        if display is not None and display.on:
            display.display(trackers)
        frame0 = frame1.copy()

    results = []
    tracklets = [tracker.tracklet for tracker in trackers]

    for tracker_nb, dets in enumerate(tracklets):
        for det in dets:
            results.append((det[0], tracker_nb, det[1][0], det[1][1], det[2], det[3]))

    results = sorted(results, key=lambda x: x[0])

    return results
