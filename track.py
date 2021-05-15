import cv2
import json
import torch
from common.utils import transform_test_CenterNet, nms
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from tracking.utils import init_trackers, compute_flow, load_base, load_extension, resize_for_network_input, gather_filenames_for_video_in_annotations
from tracking.trackers import trackers
import pickle

import matplotlib.pyplot as plt
from types import SimpleNamespace
display = SimpleNamespace()
display.on = True 
display.legends = []

if display.on:

    fig, display.ax = plt.subplots()
    plt.ion()

prop_cycle = plt.rcParams['axes.prop_cycle']
display.colors = prop_cycle.by_key()['color']

def detect_base_extension(frame, threshold, base_model, extension_model):

    frame = transform_test_CenterNet()(frame).to('cuda').unsqueeze(0)
    base_result = base_model(frame)[-1]['hm']
    extension_result = torch.sigmoid(extension_model(base_result))
    detections = nms(extension_result).gt(threshold).squeeze()

    return torch.nonzero(detections).cpu().numpy()[:, ::-1]

def detect_base(frame, threshold, base_model):

    frame = transform_test_CenterNet()(frame).to('cuda').unsqueeze(0)
    base_result = torch.sigmoid(base_model(frame)[-1]['hm'])
    detections = nms(base_result).gt(threshold).squeeze()

    return torch.nonzero(detections).cpu().numpy()[:, ::-1]

def detect_internal(reader, detector):

    detections = []

    for frame in tqdm(reader):

        detections_for_frame = detector(frame)
        if len(detections_for_frame): detections.append(detections_for_frame)
        else: detections.append(np.array([]))

    return detections

def detect_external(detections_filename, file_type='mot', nb_frames=None):

    if file_type == 'mot':
        with open(detections_filename, 'r') as f:
            detections_read = [detection.split(',') for detection in f.readlines()]
        detections_from_file = defaultdict(list)
        for detection in detections_read:
            detections_from_file[int(detection[0])].append(
                [float(detection[2]), float(detection[3])])

        detections_from_file = {k: np.array(v)
                                for k, v in detections_from_file.items()}

        detections = []

        for frame_nb in range(nb_frames):
            if frame_nb+1 in detections_from_file.keys():
                detections.append(detections_from_file[frame_nb+1])
            else:
                detections.append(np.array([]))


    elif file_type == 'pickle':
        with open(detections_filename, 'rb') as f:
            detections_read = pickle.load(f)
        
        detections = []

        for detections_for_frame in detections_read.values():
            if len(detections_for_frame): 
                detections.append(np.concatenate([detection['ct'].reshape(1,2) for detection in detections_for_frame]))
            else: detections.append(np.array([]))

    return detections 

def build_confidence_function_for_trackers(trackers, flow01):

    global legends
    confidence_functions = dict()
    if display.on: 
        display.ax.imshow(display.latest_frame_to_show)
    for tracker_nb, tracker in enumerate(trackers):
        if tracker.enabled:
            confidence_functions[tracker_nb] = tracker.build_confidence_function(flow01, tracker_nb, display)

    if display.on:
        if len(display.latest_detections): 
            display.ax.scatter(display.latest_detections[:, 0], display.latest_detections[:, 1], c='r', s=40)

        display.ax.xaxis.tick_top()
        plt.legend(handles=display.legends)
        fig.canvas.draw()
        plt.show()
        while not plt.waitforbuttonpress():
            continue
        display.ax.cla()
        display.legends = []

    return confidence_functions

def track_video(reader, detections, args, engine, state_variance, observation_variance):

    init = False
    trackers = dict()
    frame_nb = 0
    frame0, old_shape, new_shape  = next(reader)

    downsampled_shape = (new_shape[1] // args.downsampling_factor, new_shape[0] // args.downsampling_factor)
    detections_for_frame = detections[frame_nb]

    if display.on:
        display.latest_detections = detections_for_frame        
        display.latest_frame_to_show = cv2.cvtColor(cv2.resize(frame0, downsampled_shape), cv2.COLOR_BGR2RGB)

    if len(detections_for_frame):
        trackers = init_trackers(engine, detections_for_frame, frame_nb, state_variance, observation_variance, args.stop_tracking_threshold)
        init = True

    for frame_nb in tqdm(range(1,len(detections))):
        detections_for_frame = detections[frame_nb]
        frame1 = next(reader)[0]
        if display.on:
            display.latest_detections = detections_for_frame
            display.latest_frame_to_show = cv2.cvtColor(cv2.resize(frame1, downsampled_shape), cv2.COLOR_BGR2RGB)

        if not init:
            if len(detections_for_frame):
                trackers = init_trackers(engine, detections_for_frame, frame_nb, state_variance, observation_variance, args.stop_tracking_threshold)
                init = True

        else:

            new_trackers = []
            if display.on:
                flow01 = compute_flow(frame0, frame1, args.downsampling_factor)
                confidence_functions_for_trackers = build_confidence_function_for_trackers(trackers, flow01)
            if len(detections_for_frame):
                if not display.on:
                    flow01 = compute_flow(frame0, frame1, args.downsampling_factor)
                    confidence_functions_for_trackers = build_confidence_function_for_trackers(trackers, flow01)
                assigned_trackers = -np.ones(len(detections_for_frame), dtype=int)
                assignment_confidences = -np.ones(len(detections_for_frame))

                if len(confidence_functions_for_trackers):
                    for detection_nb in range(len(detections_for_frame)):

                        tracker_scores = {tracker_nb: confidence_for_tracker(detections_for_frame[detection_nb]) for tracker_nb, confidence_for_tracker in confidence_functions_for_trackers.items()}

                        tracker_ids = list(tracker_scores.keys())
                        candidate_tracker_id = tracker_ids[int(
                            np.argmax(list(tracker_scores.values())))]

                        score_for_candidate_cloud = tracker_scores[candidate_tracker_id]

                        if score_for_candidate_cloud > args.confidence_threshold:

                            if candidate_tracker_id in assigned_trackers:
                                detection_id_of_conflict = np.argwhere(
                                    assigned_trackers == candidate_tracker_id)
                                if score_for_candidate_cloud > assignment_confidences[detection_id_of_conflict]:
                                    assigned_trackers[detection_id_of_conflict] = -1
                                    assignment_confidences[detection_id_of_conflict] = -1
                                    assigned_trackers[detection_nb] = candidate_tracker_id
                                    assignment_confidences[detection_nb] = score_for_candidate_cloud
                            else:
                                assigned_trackers[detection_nb] = candidate_tracker_id
                                assignment_confidences[detection_nb] = score_for_candidate_cloud

                for detection_nb in range(len(detections_for_frame)):
                    detection = detections_for_frame[detection_nb]
                    assigned_tracker = assigned_trackers[detection_nb]
                    if assigned_tracker == -1:
                        new_trackers.append(
                            engine(frame_nb, detection, state_variance, observation_variance, stop_tracking_threshold=args.stop_tracking_threshold, algorithm=args.algorithm))
                    else:
                        trackers[assigned_tracker].update(
                            detection, flow01, frame_nb)

            for tracker in trackers:
                tracker.update_status(flow01)

            if len(new_trackers):
                trackers.extend(new_trackers)
             
        frame0 = frame1.copy()

    results = []
    tracklets = [tracker.tracklet for tracker in trackers]
    tracklets = [tracklet for tracklet in tracklets if len(
        tracklet) > args.stop_tracking_threshold]

    for tracker_nb, associated_detections in enumerate(tracklets):
        for associated_detection in associated_detections:
            results.append(
                (associated_detection[0], tracker_nb, associated_detection[1][0], associated_detection[1][1]))

    results = sorted(results, key=lambda x: x[0])

    return results

class VideoReader:

    def __init__(self, video_filename):
        self.video = cv2.VideoCapture(video_filename)

    def __next__(self):
        ret, frame = self.video.read()
        if ret: 
            return resize_for_network_input(frame)
        raise StopIteration

    def __iter__(self):
        return self
    
    def init(self):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)

def main(args):

    
    if args.detector.split('_')[0] == 'internal':
        base_model = load_base(args.base_weights)
        if args.detector.split('_')[1] == 'full':
            extension_model = load_extension(args.extension_weights, 32)

            def detector(frame): return detect_base_extension(frame, threshold=args.detection_threshold,
                                                              base_model=base_model, extension_model=extension_model)
        elif args.detector.split('_')[1] == 'base':
            def detector(frame): return detect_base(frame, threshold=args.detection_threshold,
                                                    base_model=base_model)

    state_variance = np.load(os.path.join(args.tracker_parameters_dir, 'state_variance.npy'))
    observation_variance = np.load(os.path.join(
        args.tracker_parameters_dir, 'observation_variance.npy'))

    engine = trackers[args.algorithm]

    if args.read_from == 'annotations':

        with open(args.annotation_file, 'rb') as f:
            annotations = json.load(f)

        for video in annotations['videos']:

            # video = [video_annotation for video_annotation in annotations['videos']
            #          if video_annotation['file_name'] == 'leloing__5'][0]  # debug
            filenames_for_video = gather_filenames_for_video_in_annotations(video, annotations['images'], args.data_dir)
            reader = (resize_for_network_input(cv2.imread(filename)) for filename in filenames_for_video)
            _ , old_shape, new_shape = resize_for_network_input(cv2.imread(filenames_for_video[0])) 
            ratio_y = old_shape[0] / (new_shape[0] // args.downsampling_factor)
            ratio_x = old_shape[1] / (new_shape[1] // args.downsampling_factor)

            output_filename = os.path.join(
                args.output_dir, video['file_name']+'.txt')
            output_file = open(output_filename, 'w')

            if args.detector.split('_')[0] == 'internal':
                detections = detect_internal(reader, detector)
            else:
                detections_filename = os.path.join(args.external_detections_dir, video['file_name']+'.txt')
                detections = detect_external(detections_filename=detections_filename, file_type=args.detector.split('_')[1], nb_frames=len(filenames_for_video))
                detections_resized = []
                for detection in detections:
                    if len(detection):
                        detections_resized.append(
                            np.array([1/ratio_x, 1/ratio_y])*detection)
                    else:
                        detections_resized.append(detection)
                detections = detections_resized


            results = track_video(
                reader, detections, args, engine, state_variance, observation_variance)

            for result in results:
                output_file.write('{},{},{},{},{},{},{},{},{},{}\n'.format(result[0]+1,
                                                                        result[1]+1,
                                                                        ratio_x *
                                                                        result[2],
                                                                        ratio_y *
                                                                        result[3],
                                                                        -1,
                                                                        -1,
                                                                        1,
                                                                        -1,
                                                                        -1,
                                                                        -1))

            output_file.close()

    elif args.read_from == 'folder':
        video_filenames = [video_filename for video_filename in os.listdir(args.data_dir) if video_filename.endswith('.mp4')]

        for video_filename in video_filenames: 

            reader = VideoReader(os.path.join(args.data_dir,video_filename))

            output_filename = os.path.join(
                args.output_dir, video_filename.split('.')[0] +'.txt')
            output_file = open(output_filename, 'w')

            _ , old_shape, new_shape = next(reader)
            reader.init()
            ratio_y = old_shape[0] / (new_shape[0] // args.downsampling_factor)
            ratio_x = old_shape[1] / (new_shape[1] // args.downsampling_factor)


            if args.detector.split('_')[0] == 'internal':
                detections = detect_internal(reader, detector)
            else:
                detections_filename = os.path.join(args.external_detections_dir, video_filename.split('.')[0]+'.pickle')
                detections = detect_external(detections_filename=detections_filename, file_type=args.detector.split('_')[1], nb_frames=None)
                detections_resized = []
                for detection in detections:
                    if len(detection):
                        detections_resized.append(
                            np.array([1/ratio_x, 1/ratio_y])*detection)
                    else:
                        detections_resized.append(detection)
                detections = detections_resized

            reader.init()


            results = track_video(
                reader, detections, args, engine, state_variance, observation_variance)

            for result in results:
                output_file.write('{},{},{},{},{},{},{},{},{},{}\n'.format(result[0]+1,
                                                                        result[1]+1,
                                                                        ratio_x *
                                                                        result[2],
                                                                        ratio_y *
                                                                        result[3],
                                                                        -1,
                                                                        -1,
                                                                        1,
                                                                        -1,
                                                                        -1,
                                                                        -1))

            output_file.close()




            
        





if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Tracking')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--annotation_file', type=str)
    parser.add_argument('--stop_tracking_threshold', type=float, default=5)
    parser.add_argument('--detection_threshold', type=float, default=0.33)
    parser.add_argument('--confidence_threshold', type=float, default=0.2)
    parser.add_argument('--base_weights', type=str)
    parser.add_argument('--extension_weights', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--downsampling_factor', type=int)
    parser.add_argument('--external_detections_dir', type=str)
    parser.add_argument('--detector', type=str ,default='internal_base')
    parser.add_argument('--algorithm', type=str, default='Kalman')
    parser.add_argument('--read_from',type=str)
    parser.add_argument('--tracker_parameters_dir',type=str)
    args = parser.parse_args()

    main(args)
