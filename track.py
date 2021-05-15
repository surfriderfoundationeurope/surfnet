import cv2
import json
import numpy as np
import os
from tqdm import tqdm
from tracking.utils import init_trackers, compute_flow, load_base, load_extension, resize_for_network_input, gather_filenames_for_video_in_annotations, detect_base, detect_base_extension, detect_external, detect_internal, VideoReader
from tracking.trackers import trackers
import matplotlib.pyplot as plt


class Display:

    def __init__(self, on):
        self.on = on
        self.fig, self.ax = plt.subplots()
        plt.ion()
        self.colors =  plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.legends = []

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
            plt.show()
            while not plt.waitforbuttonpress():
                continue
            self.ax.cla()
            self.legends = []

    def update_detections_and_frame(self, latest_detections, frame):
        self.latest_detections = latest_detections
        self.latest_frame_to_show = cv2.cvtColor(cv2.resize(frame, self.display_shape), cv2.COLOR_BGR2RGB)

display = Display(on=False)

def build_confidence_function_for_trackers(trackers, flow01):

    confidence_functions = dict()
    for tracker_nb, tracker in enumerate(trackers):
        if tracker.enabled:
            confidence_functions[tracker_nb] = tracker.build_confidence_function(flow01)

    return confidence_functions

def track_video(reader, detections, args, engine, state_variance, observation_variance):

    init = False
    trackers = dict()
    frame_nb = 0
    frame0, _ , new_shape  = next(reader)
    detections_for_frame = detections[frame_nb]

    if display.on: 
        display.display_shape = (new_shape[1] // args.downsampling_factor, new_shape[0] // args.downsampling_factor)
        display.update_detections_and_frame(detections_for_frame, frame0)


    if len(detections_for_frame):
        trackers = init_trackers(engine, detections_for_frame, frame_nb, state_variance, observation_variance, args.stop_tracking_threshold)
        init = True

    if display.on: display.display(trackers)

    for frame_nb in tqdm(range(1,len(detections))):

        detections_for_frame = detections[frame_nb]
        frame1 = next(reader)[0]
        if display.on: display.update_detections_and_frame(detections_for_frame, frame1)

        if not init:
            if len(detections_for_frame):
                trackers = init_trackers(engine, detections_for_frame, frame_nb, state_variance, observation_variance, args.stop_tracking_threshold)
                init = True

        else:

            new_trackers = []
            flow01 = compute_flow(frame0, frame1, args.downsampling_factor)

            if len(detections_for_frame):
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
                            engine(frame_nb, detection, state_variance, observation_variance, stop_tracking_threshold=args.stop_tracking_threshold))
                    else:
                        trackers[assigned_tracker].update(
                            detection, flow01, frame_nb)

            for tracker in trackers:
                tracker.update_status(flow01)

            if len(new_trackers):
                trackers.extend(new_trackers)

        if display.on: display.display(trackers)
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

            video = [video_annotation for video_annotation in annotations['videos']
                     if video_annotation['file_name'] == 'leloing__5'][0]  # debug
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
