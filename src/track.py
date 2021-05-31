import cv2
import json
import numpy as np
import os
from tqdm import tqdm
from tracking.utils import init_trackers, gather_filenames_for_video_in_annotations, detect_base, detect_base_extension, detect_external, detect_internal
from common.opencv_tools import IterableFrameReader
from common.flow_tools import compute_flow
from common.utils import load_base, load_extension
from tracking.trackers import trackers, DetectionFreeTracker
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('TkAgg')
import pickle

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
            plt.title('Raw count: {}'.format(len(trackers)))
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
    frame0 = next(reader)
    detections_for_frame = detections[frame_nb]

    if display.on: 
        display.display_shape = (reader.output_shape[0] // args.downsampling_factor, reader.output_shape[1] // args.downsampling_factor)
        display.update_detections_and_frame(detections_for_frame, frame0)

    if len(detections_for_frame):
        trackers = init_trackers(engine, detections_for_frame, frame_nb, state_variance, observation_variance, args.stop_tracking_threshold)
        init = True

    if display.on: display.display(trackers)

    for frame_nb in tqdm(range(1,len(detections))):

        detections_for_frame = detections[frame_nb]
        frame1 = next(reader)
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
    tracklets = [tracker.tracklet for tracker in trackers if not tracker.unstable]
    tracklets = [tracklet for tracklet in tracklets if len(tracklet) > args.stop_tracking_threshold]
    

    for tracker_nb, associated_detections in enumerate(tracklets):
        for associated_detection in associated_detections:
            results.append(
                (associated_detection[0], tracker_nb, associated_detection[1][0], associated_detection[1][1]))

    results = sorted(results, key=lambda x: x[0])
 
    return results

def track_video_2(reader, heatmaps, args, engine, state_variance, observation_variance):

    heatmaps = heatmaps[:200]
    heatmap = heatmaps[0]
    tracker = DetectionFreeTracker(heatmap, jump_probability=0.1, state_variance=state_variance, observation_variance=observation_variance, num_samples=500)
    display_shape = (reader.output_shape[0] , reader.output_shape[1])

    frame0 = next(reader)   

    for frame_nb in tqdm(range(1,len(heatmaps))): 
        heatmap = heatmaps[frame_nb] 

        frame1 = next(reader)
        flow01 = compute_flow(frame0, frame1, args.downsampling_factor)
        tracker.update(heatmap, flow01)

        frame0 = frame1.copy()
        

    reader.init()
    for frame_nb in range(len(heatmaps)):

        frame_for_samples = np.zeros(display_shape[::-1])
        # frame_for_mean = np.zeros(display_shape[::-1])
        # mean_of_samples = args.downsampling_factor * np.mean(tracker.samples[frame_nb], axis=0)
        # print((int(mean_of_samples[0]),int(mean_of_samples[1])))
        # cv2.circle(frame_for_mean, (int(mean_of_samples[0]),int(mean_of_samples[1])), radius=5, color=(255,0,0))

        for sample in tracker.samples[frame_nb]:
            cv2.circle(frame_for_samples, (args.downsampling_factor*sample[0],args.downsampling_factor*sample[1]), radius=1, color=(255,0,0))

        cv2.imshow('frame',next(reader))
        cv2.imshow('heatmap',cv2.resize(heatmaps[frame_nb], display_shape))
        cv2.imshow('samples',frame_for_samples)
        # cv2.imshow('mean_of_samples',frame_for_mean)
        cv2.waitKey(0)


    return tracker.samples

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

            input_shape = cv2.imread(filenames_for_video[0]).shape[::-1]
            if args.output_shape is None:
                h, w = input_shape
                new_h = (h | 31) + 1
                new_w = (w | 31) + 1
                output_shape = (new_w, new_h)
            else: 
                output_shape = args.output_shape

            reader = (cv2.resize(cv2.imread(filename), output_shape) for filename in filenames_for_video)
            ratio_y = input_shape[0] / (output_shape[0] // args.downsampling_factor)
            ratio_x = input_shape[1] / (output_shape[1] // args.downsampling_factor)

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

            with open(output_filename.split('.')[0]+'.pickle','wb') as f:
                pickle.dump(detections, f)

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

            reader = IterableFrameReader(os.path.join(args.data_dir,video_filename), skip_frames=args.skip_frames, output_shape=args.output_shape)

            output_filename = os.path.join(
                args.output_dir, video_filename.split('.')[0] +'.txt')
            output_file = open(output_filename, 'w')

            input_shape = reader.input_shape
            output_shape = reader.output_shape
            ratio_y = input_shape[0] / (output_shape[0] // args.downsampling_factor)
            ratio_x = input_shape[1] / (output_shape[1] // args.downsampling_factor)


            if args.detector.split('_')[0] == 'internal':
                detections, heatmaps = detect_internal(reader, detector)
                detections_save_folder = os.path.join(args.output_dir,'detections')
                heatmaps_save_folder = os.path.join(args.output_dir,'heatmaps')
                os.mkdir(detections_save_folder)
                os.mkdir(heatmaps_save_folder)
                with open(os.path.join(detections_save_folder,video_filename.split('.')[0]+'.pickle'),'wb') as f:
                    pickle.dump(detections,f)
                with open(os.path.join(heatmaps_save_folder,video_filename.split('.')[0]+'.pickle'),'wb') as f:
                    pickle.dump(heatmaps,f)
            else:
                detections_filename = os.path.join(args.external_detections_dir, 'detections', video_filename.split('.')[0]+'.pickle')
                heatmaps_filename = os.path.join(args.external_detections_dir, 'heatmaps', video_filename.split('.')[0]+'.pickle')

                detections, heatmaps = detect_external(detections_filename=detections_filename, heatmaps_filename=heatmaps_filename, file_type=args.detector.split('_')[1], nb_frames=None)
                
                if not args.detector.split('_')[1] == 'simplepickle':
                    detections_resized = []
                    for detection in detections:
                        if len(detection):
                            detections_resized.append(
                                detection/args.downsampling_factor)
                        else:
                            detections_resized.append(detection)
                    detections = detections_resized

            if args.version == 'from_detections': 
                results = track_video(reader, detections, args, engine, state_variance, observation_variance)
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

            else: 
                results = track_video_2(reader, heatmaps, args, engine, state_variance, observation_variance)
                with open('samples.pickle','wb') as f:
                    pickle.dump(results,f)

                



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
    parser.add_argument('--skip_frames',type=int,default=0)
    parser.add_argument('--output_w',type=int,default=None)
    parser.add_argument('--output_h',type=int,default=None)
    parser.add_argument('--version',type=str,default='from_detections')

    args = parser.parse_args()

    if args.output_w is not None:
        args.output_shape = (args.output_w, args.output_h)
    else: 
        args.output_shape = None

    main(args)
