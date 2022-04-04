import cv2
import numpy as np
import os
from plasticorigins.detection.detect import detect
from plasticorigins.tracking.utils import in_frame, init_trackers
from plasticorigins.tools.optical_flow import compute_flow
from plasticorigins.tracking.trackers import get_tracker
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import torch


class Display:
    def __init__(self, on, interactive=True):
        self.on = on
        self.fig, self.ax = plt.subplots()
        self.interactive = interactive
        if interactive:
            plt.ion()
        self.colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.legends = []
        self.plot_count = 0

    def display(self, trackers):

        something_to_show = False
        for tracker_nb, tracker in enumerate(trackers):
            if tracker.enabled:
                tracker.fill_display(self, tracker_nb)
                something_to_show = True

        self.ax.imshow(self.latest_frame_to_show)

        if len(self.latest_detections):
            self.ax.scatter(self.latest_detections[:, 0], self.latest_detections[:, 1], c='r', s=40)

        if something_to_show:
            self.ax.xaxis.tick_top()
            plt.legend(handles=self.legends)
            self.fig.canvas.draw()
            if self.interactive:
                plt.show()
                while not plt.waitforbuttonpress():
                    continue
            else:
                plt.savefig(os.path.join('plots',str(self.plot_count)))
            self.ax.cla()
            self.legends = []
            self.plot_count+=1

    def update_detections_and_frame(self, latest_detections, frame):
        self.latest_detections = latest_detections
        self.latest_frame_to_show = cv2.cvtColor(cv2.resize(frame, self.display_shape), cv2.COLOR_BGR2RGB)


def build_confidence_function_for_trackers(trackers, flow01):
    tracker_nbs = []
    confidence_functions = []
    for tracker_nb, tracker in enumerate(trackers):
        if tracker.enabled:
            tracker_nbs.append(tracker_nb)
            confidence_functions.append(tracker.build_confidence_function(flow01))
    return tracker_nbs, confidence_functions

def associate_detections_to_trackers(detections_for_frame, trackers, flow01, confidence_threshold):
    tracker_nbs, confidence_functions = build_confidence_function_for_trackers(trackers, flow01)
    assigned_trackers = [None]*len(detections_for_frame)
    if len(tracker_nbs):
        cost_matrix = np.zeros(shape=(len(detections_for_frame),len(tracker_nbs)))
        for detection_nb, detection in enumerate(detections_for_frame):
            for tracker_id, confidence_function in enumerate(confidence_functions):
                score = confidence_function(detection)
                if score > confidence_threshold:
                    cost_matrix[detection_nb,tracker_id] = score
                else:
                    cost_matrix[detection_nb,tracker_id] = 0
        row_inds, col_inds = linear_sum_assignment(cost_matrix,maximize=True)
        for row_ind, col_ind in zip(row_inds, col_inds):
            if cost_matrix[row_ind,col_ind] > confidence_threshold: assigned_trackers[row_ind] = tracker_nbs[col_ind]

    return assigned_trackers

def track_video(reader, detections, args, engine, transition_variance, observation_variance, display):
    init = False
    trackers = dict()
    frame_nb = 0
    frame0 = next(reader)
    detections_for_frame = next(detections)

    max_distance = euclidean(reader.output_shape, np.array([0,0]))
    delta = 0.05*max_distance

    if display is not None and display.on:

        display.display_shape = (reader.output_shape[0] // args.downsampling_factor, reader.output_shape[1] // args.downsampling_factor)
        display.update_detections_and_frame(detections_for_frame, frame0)

    if len(detections_for_frame):
        trackers = init_trackers(engine, detections_for_frame, frame_nb, transition_variance, observation_variance, delta)
        init = True

    if display is not None and display.on: display.display(trackers)

    for frame_nb, (frame1, detections_for_frame) in enumerate(zip(reader, detections), start=1):

        if display is not None and display.on:
            display.update_detections_and_frame(detections_for_frame, frame1)

        if not init:
            if len(detections_for_frame):
                trackers = init_trackers(engine, detections_for_frame, frame_nb, transition_variance, observation_variance, delta)
                init = True

        else:

            new_trackers = []
            flow01 = compute_flow(frame0, frame1, args.downsampling_factor)

            if len(detections_for_frame):

                assigned_trackers = associate_detections_to_trackers(detections_for_frame, trackers,
                                                                     flow01, args.confidence_threshold)

                for detection, assigned_tracker in zip(detections_for_frame, assigned_trackers):
                    if in_frame(detection, flow01.shape[:-1]):
                        if assigned_tracker is None :
                            new_trackers.append(engine(frame_nb, detection, transition_variance, observation_variance, delta))
                        else:
                            trackers[assigned_tracker].update(detection, flow01, frame_nb)

            for tracker in trackers:
                tracker.update_status(flow01)

            if len(new_trackers):
                trackers.extend(new_trackers)

        if display is not None and display.on:
            display.display(trackers)
        frame0 = frame1.copy()


    results = []
    tracklets = [tracker.tracklet for tracker in trackers]

    for tracker_nb, associated_detections in enumerate(tracklets):
        for associated_detection in associated_detections:
            results.append((associated_detection[0], tracker_nb, associated_detection[1][0], associated_detection[1][1]))

    results = sorted(results, key=lambda x: x[0])

    return results
