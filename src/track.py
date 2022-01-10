import cv2
import numpy as np
import os
from detection.detect import detect
from tracking.utils import in_frame, init_trackers, get_detections_for_video, write_tracking_results_to_file
from tools.video_readers import IterableFrameReader
from tools.optical_flow import compute_flow
from tools.misc import load_model
from tracking.trackers import get_tracker
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.optimize import linear_sum_assignment
import torch 
class Display:

    def __init__(self, on, interactive=True):
        self.on = on
        self.fig, self.ax0 = plt.subplots(figsize=(20,10))
        self.interactive = interactive
        if interactive:
            plt.ion()
        self.colors =  ['blue','red'] #plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.legends = []
        self.plot_count = 0
        self.flow = None
        
    def display(self, trackers):

        something_to_show = False
        for tracker_nb, tracker in enumerate(trackers): 
            if tracker.enabled:
                tracker.fill_display(self, tracker_nb)
                something_to_show = True

        self.ax0.imshow(self.latest_frame_to_show)
        # self.ax1.imshow(self.latest_frame_to_show)

        # if len(self.latest_detections): 
        #     self.ax0.scatter(self.latest_detections[:, 0], self.latest_detections[:, 1], c='r', s=40)
            
        if something_to_show: 
            self.ax0.xaxis.tick_top()
            if self.flow is not None:
                magnitude, angle = cv2.cartToPolar(self.flow[..., 0], self.flow[..., 1])
      
                # Sets image hue according to the optical flow 
                # direction
                self.mask[..., 0] = angle * 180 / np.pi / 2
                
                # Sets image value according to the optical flow
                # magnitude (normalized)
                self.mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

                rgb = cv2.cvtColor(self.mask, cv2.COLOR_HSV2BGR)
            
                # Opens a new window and displays the output frame
                cv2.imwrite(os.path.join('plots','flow_'+str(self.plot_count)+'.png'),rgb)

            # plt.legend(handles=self.legends)
            self.fig.canvas.draw()
            self.ax0.set_axis_off()
            # self.ax1.set_axis_off()
            plt.autoscale(True)
            plt.tight_layout()
            if self.interactive: 
                plt.show()
                while not plt.waitforbuttonpress():
                    continue
            else:
                plt.savefig(os.path.join('plots',str(self.plot_count)))
            self.ax0.cla()
            # self.ax1.cla()
            self.legends = []
            self.plot_count+=1

    def update_detections_and_frame(self, latest_detections, frame):
        self.latest_detections = latest_detections
        self.latest_frame_to_show = cv2.cvtColor(cv2.resize(frame, self.display_shape), cv2.COLOR_BGR2RGB)
    
    def update_flow(self, flow):
        self.flow = flow
        self.mask = np.zeros_like(self.latest_frame_to_show)
  
        # Sets image saturation to maximum
        self.mask[..., 1] = 255

display = None 

def build_confidence_function_for_trackers(trackers, flow01):
    tracker_nbs = []
    confidence_functions = []
    for tracker_nb, tracker in enumerate(trackers):
        if tracker.enabled:
            tracker_nbs.append(tracker_nb)
            confidence_functions.append(tracker.build_confidence_function(flow01))
    return tracker_nbs, confidence_functions

def associate_detections_to_trackers(detections_for_frame, trackers, flow01):
    tracker_nbs, confidence_functions = build_confidence_function_for_trackers(trackers, flow01)
    assigned_trackers = [None]*len(detections_for_frame)
    if len(tracker_nbs):
        cost_matrix = np.zeros(shape=(len(detections_for_frame),len(tracker_nbs)))
        for detection_nb, detection in enumerate(detections_for_frame):
            for tracker_id, confidence_function in enumerate(confidence_functions):
                score = confidence_function(detection)
                if score > args.confidence_threshold:
                    cost_matrix[detection_nb,tracker_id] = score
                else:
                    cost_matrix[detection_nb,tracker_id] = 0
        row_inds, col_inds = linear_sum_assignment(cost_matrix,maximize=True)
        for row_ind, col_ind in zip(row_inds, col_inds):
            if cost_matrix[row_ind,col_ind] > args.confidence_threshold: assigned_trackers[row_ind] = tracker_nbs[col_ind]

    return assigned_trackers
    
def track_video(reader, detections, args, engine, transition_variance, observation_variance):
    init = False
    trackers = dict()
    frame_nb = 0
    frame0 = next(reader)
    detections_for_frame = next(detections)

    max_distance = euclidean(reader.output_shape, np.array([0,0]))
    delta = 0.05*max_distance

    if display.on: 
    
        display.display_shape = (reader.output_shape[0] // args.downsampling_factor, reader.output_shape[1] // args.downsampling_factor)
        display.update_detections_and_frame(detections_for_frame, frame0)

    if len(detections_for_frame):
        trackers = init_trackers(engine, detections_for_frame, frame_nb, transition_variance, observation_variance, delta)
        init = True

    if display.on: display.display(trackers)

    for frame_nb, (frame1, detections_for_frame) in enumerate(zip(reader, detections), start=1):

        if display.on: display.update_detections_and_frame(detections_for_frame, frame1)

        if not init:
            if len(detections_for_frame):
                trackers = init_trackers(engine, detections_for_frame, frame_nb, transition_variance, observation_variance, delta)
                init = True

        else:

            new_trackers = []
            flow01 = compute_flow(frame0, frame1, args.downsampling_factor)

            if len(detections_for_frame):

                assigned_trackers = associate_detections_to_trackers(detections_for_frame, trackers, flow01)

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

        if display.on: display.display(trackers)
        frame0 = frame1.copy()


    results = []
    tracklets = [tracker.tracklet for tracker in trackers]
    
    for tracker_nb, associated_detections in enumerate(tracklets):
        for associated_detection in associated_detections:
            results.append((associated_detection[0], tracker_nb, associated_detection[1][0], associated_detection[1][1]))

    results = sorted(results, key=lambda x: x[0])
 
    return results

def main(args):


    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else: 
        device = args.device
    device = torch.device(device)


    engine = get_tracker('EKF')

    print('---Loading model...')            
    model = load_model(arch=args.arch, model_weights=args.model_weights, device=device)
    print('Model loaded.')

    detector = lambda frame: detect(frame, threshold=args.detection_threshold, model=model)

    transition_variance = np.load(os.path.join(args.noise_covariances_path, 'transition_variance.npy'))
    observation_variance = np.load(os.path.join(args.noise_covariances_path, 'observation_variance.npy'))

    video_filenames = [video_filename for video_filename in os.listdir(args.data_dir) if video_filename.endswith('.mp4')]

    for video_filename in video_filenames: 
        print(f'---Processing {video_filename}')        
        reader = IterableFrameReader(video_filename=os.path.join(args.data_dir, video_filename), 
                                     skip_frames=args.skip_frames, 
                                     output_shape=args.output_shape,
                                     progress_bar=True,
                                     preload=args.preload_frames)


        input_shape = reader.input_shape
        output_shape = reader.output_shape
        ratio_y = input_shape[0] / (output_shape[0] // args.downsampling_factor)
        ratio_x = input_shape[1] / (output_shape[1] // args.downsampling_factor)

        print('Detecting...')
        detections = get_detections_for_video(reader, detector, batch_size=args.detection_batch_size, device=device)

        print('Tracking...')
        results = track_video(reader, iter(detections), args, engine, transition_variance, observation_variance)

        output_filename = os.path.join(args.output_dir, video_filename.split('.')[0] +'.txt')
        write_tracking_results_to_file(results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='Tracking')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--detection_threshold', type=float, default=0.3)
    parser.add_argument('--confidence_threshold', type=float, default=0.2)
    parser.add_argument('--model_weights', type=str, default=None)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--downsampling_factor', type=int, default=1)
    parser.add_argument('--noise_covariances_path',type=str)
    parser.add_argument('--skip_frames',type=int,default=0)
    parser.add_argument('--output_shape',type=str,default='960,544')
    parser.add_argument('--arch', type=str, default='dla_34')
    parser.add_argument('--display', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--detection_batch_size',type=int,default=1)
    parser.add_argument('--preload_frames', action='store_true', default=False)
    args = parser.parse_args()

    if args.display == 0: 
        display = Display(on=False, interactive=True)
    elif args.display == 1:
        display = Display(on=True, interactive=True)
    elif args.display == 2:
        display = Display(on=True, interactive=False)

    args.output_shape = tuple(int(s) for s in args.output_shape.split(','))

    main(args)
