import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean

from plasticorigins.tools.optical_flow import compute_flow
from plasticorigins.tracking.utils import in_frame


def init_trackers(
    engine,
    detections,
    confs,
    labels,
    frame_nb,
    state_variance,
    observation_variance,
    delta,
):
    """Initializes the trackers based on detections"""
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


def build_confidence_function_for_trackers(trackers, flow01):
    tracker_nbs = []
    confidence_functions = []
    for tracker_nb, tracker in enumerate(trackers):
        if tracker.enabled:
            tracker_nbs.append(tracker_nb)
            confidence_functions.append(tracker.build_confidence_function(flow01))
    return tracker_nbs, confidence_functions


def associate_detections_to_trackers(
    detections_for_frame, confs, labels, trackers, flow01, confidence_threshold
):
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


def interpret_detection(detections_for_frame, downsampling_factor, is_yolo=False):
    """normalizes the detections depending whether they come from centernet or yolo"""
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
    reader,
    detections,
    args,
    engine,
    transition_variance,
    observation_variance,
    display,
    is_yolo=False,
):
    """
    Original version. Expects detections in the format list[np.array([[xcenter, ycenter], ...]), ...]
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
