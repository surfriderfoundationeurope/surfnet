from threading import Condition
import cv2
import json

from torch.functional import align_tensors
from base.centernet.models import create_model as create_base
from common.utils import load_my_model
from extension.models import SurfNet
import torch
import matplotlib.pyplot as plt
from common.utils import transform_test_CenterNet, nms
from synthetic_videos.flow_tools import flow_opencv_dense
import numpy as np
import os
from scipy.stats import multivariate_normal
from torchvision.transforms.functional import resize
from tqdm import tqdm
from collections import defaultdict
import matplotlib.patches as mpatches


verbose = True
latest_detections = None
latest_frame_to_show = None
latest_image = None
show_detections_only = False
frame_nb_for_plot = None
frames = []
legends = []
from pykalman import KalmanFilter

if verbose:

    fig, ax = plt.subplots()
    # fig.canvas.mpl_connect('key_press_event', press)
    plt.ion()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


def exp_and_normalise(lw):
    w = np.exp(lw - lw.max())
    return w / w.sum()

def in_frame(position, shape):

    shape_x = shape[1]
    shape_y = shape[0]
    x = position[0]
    y = position[1]

    return x > 0 and x < shape_x and y > 0 and y < shape_y

class GaussianMixture(object):
    def __init__(self, means, covariance, weights):
        self.components = [multivariate_normal(
            mean=mean, cov=covariance) for mean in means]
        self.weights = weights

    def pdf(self, x):
        result = 0
        for weight, component in zip(self.weights, self.components):
            result += weight*component.pdf(x)
        return result

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def cdf(self, x):
        result = 0
        for weight, component in zip(self.weights, self.components):
            result += weight*component.cdf(x)
        return result

def confidence_from_multivariate_distribution(coord, distribution):

    delta = 3
    x = coord[0]
    y = coord[1]
    right_top = np.array([x+delta, y+delta])
    left_low = np.array([x-delta, y-delta])
    right_low = np.array([x+delta, y-delta])
    left_top = np.array([x-delta, y+delta])

    return distribution.cdf(right_top) \
        - distribution.cdf(right_low) \
        - distribution.cdf(left_top) \
        + distribution.cdf(left_low)

class Tracker:

    def __init__(self, frame_nb, X0, state_variance, observation_variance, stop_tracking_threshold=5, algorithm='Kalman'):

        self.state_covariance = np.diag(state_variance)
        self.observation_covariance = np.diag(observation_variance)
        self.algorithm = algorithm

        if self.algorithm =='SMC' :
            self.particles, self.normalized_weights = self.init_particles(
            X0, n_particles=20)
        else: 
                
            self.filter = KalmanFilter(initial_state_mean=X0, 
                                       initial_state_covariance=self.observation_covariance, 
                                       transition_matrices=np.eye(2), 
                                       transition_covariance=self.state_covariance,
                                       observation_matrices=np.eye(2))

            self.filtered_state_mean = X0
            self.filtered_state_covariance = self.observation_covariance

        self.updated = False
        self.countdown = 0
        self.enabled = True
        self.stop_tracking_threshold = stop_tracking_threshold
        self.tracklet = [(frame_nb, X0)]

    def init_particles(self, X0, n_particles):
        particles = multivariate_normal(
            X0, cov=self.observation_covariance).rvs(n_particles)
        normalized_weights = np.ones(n_particles)/n_particles
        return particles, normalized_weights

    def SMC_state_transition(self, state, flow):

        mean = state + \
            flow[max(0, int(state[1])),
                 max(0, int(state[0])), :]
        cov = np.diag(self.state_covariance)
        return multivariate_normal(mean, cov)
    
    def SMC_observation(self, state):

        return multivariate_normal(state, self.observation_covariance)

    def move_particles(self, flow):
        new_particles = []
        for particle in self.particles:
            new_particle = self.SMC_state_transition(particle, flow).rvs(1)
            if in_frame(new_particle, flow.shape[:-1]):
                new_particles.append(new_particle)
        if len(new_particles):
            self.particles = np.array(new_particles)
        else:
            self.enabled = False

    def importance_reweighting(self, observation):
        log_weights_unnormalized = np.zeros(len(self.particles))
        for particle_nb, particle in enumerate(self.particles):
            log_weights_unnormalized[particle_nb] = self.SMC_observation(
                particle).logpdf(observation)
        self.normalized_weights = exp_and_normalise(log_weights_unnormalized)

    def resample(self):
        resampling_indices = np.random.choice(
            a=len(self.particles), p=self.normalized_weights, size=len(self.particles))
        self.particles = self.particles[resampling_indices]

    def update_SMC(self, observation, flow):

        self.resample()
        self.move_particles(flow)
        if observation is not None: 
            self.importance_reweighting(observation)
        else: 
            self.normalized_weights = np.ones(
                len(self.particles))/len(self.particles)


    def update_Kalman(self, observation, flow):
        transition_offset = flow[max(0, int(self.filtered_state_mean[1])),
                 max(0, int(self.filtered_state_mean[0])), :]

        self.filtered_state_mean, self.filtered_state_covariance = self.filter.filter_update(self.filtered_state_mean, 
                                                                                             self.filtered_state_covariance, 
                                                                                             observation=observation,
                                                                                             transition_offset=transition_offset)
        if not in_frame(self.filtered_state_mean,flow.shape[:-1]): self.enabled=False

    def update(self, observation, flow, frame_nb):
        if self.algorithm == 'SMC': 
            self.update_SMC(observation, flow)
        else: 
            self.update_Kalman(observation, flow)

        self.tracklet.append((frame_nb, observation))
        self.updated = True

    def update_status(self, flow):
        if self.enabled and not self.updated:
            self.countdown += 1
            if self.algorithm == 'SMC':
                self.update_SMC(None, flow)
            else: 
                self.update_Kalman(None, flow)
        else:
            self.countdown = 0
        self.updated = False

        # if self.countdown > self.stop_tracking_threshold:
        #     self.enabled = False
        # if len(self.particles) < 10:
        #     self.enabled = False

    def confidence_function_SMC(self, flow, tracker_nb=None, nb_new_particles=5):
        global legends
        new_particles = []
        new_weights = []

        for particle, normalized_weight in zip(self.particles, self.normalized_weights):
            new_particles_for_particle = self.SMC_state_transition(
                particle, flow).rvs(nb_new_particles)

            new_particles_for_particle = [
                particle for particle in new_particles_for_particle if in_frame(particle, flow.shape[:-1])]

            if len(new_particles_for_particle):
                new_particles.extend(new_particles_for_particle)
                new_weights.extend([normalized_weight/len(new_particles_for_particle)] *
                                len(new_particles_for_particle))

        new_particles = np.array(new_particles)

        if verbose: 
            ax.scatter(new_particles[:,0], new_particles[:,1], s=5, c=colors[tracker_nb])
            legends.append(mpatches.Patch(color=colors[tracker_nb], label=self.countdown))

        return GaussianMixture(new_particles, self.observation_covariance, new_weights)

    def confidence_function_Kalman(self, flow, tracker_nb=None):
        global legends
        transition_offset = flow[max(0, int(self.filtered_state_mean[1])), max(0, int(self.filtered_state_mean[0])), :]

        filtered_state_mean, filtered_state_covariance = self.filter.filter_update(self.filtered_state_mean, 
                                                                                             self.filtered_state_covariance, 
                                                                                             observation=None,
                                                                                             transition_offset=transition_offset)

        distribution = multivariate_normal(filtered_state_mean, filtered_state_covariance)

        if verbose: 
            yy, xx = np.mgrid[0:flow.shape[0]:1, 0:flow.shape[1]:1]
            pos = np.dstack((xx, yy))            
            ax.contour(distribution.pdf(pos), colors=colors[tracker_nb])
            legends.append(mpatches.Patch(color=colors[tracker_nb], label=self.countdown))

        return distribution

    def build_confidence_function(self, flow, tracker_nb=None):
        if self.algorithm == 'SMC':
            distribution = self.confidence_function_SMC(flow, tracker_nb)
        else:
            distribution = self.confidence_function_Kalman(flow, tracker_nb)
        
        return lambda coord: confidence_from_multivariate_distribution(coord, distribution)

def init_trackers(detections, frame_nb, state_variance, observation_variance, algorithm, stop_tracking_threshold):
    trackers = []

    for detection in detections:
        tracker_for_detection = Tracker(frame_nb,
                                            detection, state_variance, observation_variance, stop_tracking_threshold=stop_tracking_threshold, algorithm=algorithm)
        trackers.append(tracker_for_detection)

    return trackers


def compute_flow(frame0, frame1, downsampling_factor):
    h, w = frame0.shape[:-1]

    new_h = h // downsampling_factor
    new_w = w // downsampling_factor

    frame0 = cv2.resize(frame0, (new_w, new_h))
    frame1 = cv2.resize(frame1, (new_w, new_h))

    flow01 = flow_opencv_dense(frame0, frame1)
    # if verbose:
    #     flow01 = np.ones_like(flow01)
    #     flow01[:,:,0] = 5
    #     flow01[:,:,1] = 100
    return flow01


def load_extension(extension_weights, intermediate_layer_size=32):
    extension_model = SurfNet(intermediate_layer_size)
    extension_model.load_state_dict(torch.load(extension_weights))
    for param in extension_model.parameters():
        param.requires_grad = False
    extension_model.to('cuda')
    extension_model.eval()
    return extension_model


def load_base(base_weights):
    base_model = create_base('dla_34', heads={'hm': 1, 'wh': 2}, head_conv=256)
    base_model = load_my_model(base_model, base_weights)
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.to('cuda')
    base_model.eval()
    return base_model


def detect_base_extension(frame, threshold, base_model, extension_model):

    frame = transform_test_CenterNet()(frame).to('cuda').unsqueeze(0)
    base_result = base_model(frame)[-1]['hm']
    extension_result = torch.sigmoid(extension_model(base_result))
    detections = nms(extension_result).gt(threshold).squeeze()
    detections_array = torch.nonzero(detections).cpu().numpy()[:, ::-1]

    if verbose:
        global latest_image
        global latest_detection
        image = np.transpose(
            resize(frame, detections.shape).squeeze().cpu().numpy(), axes=[1, 2, 0])
        image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
        latest_image = image
        latest_detection = detections_array

    return detections_array


def detect_base(frame, threshold, base_model):

    frame = transform_test_CenterNet()(frame).to('cuda').unsqueeze(0)
    base_result = torch.sigmoid(base_model(frame)[-1]['hm'])
    detections = nms(base_result).gt(threshold).squeeze()

    return torch.nonzero(detections).cpu().numpy()[:, ::-1]


def read_and_resize(filename):
    frame = cv2.imread(filename)
    h, w = frame.shape[:-1]
    new_h = (h | 31) + 1
    new_w = (w | 31) + 1
    frame = cv2.resize(frame, (new_w, new_h))
    new_shape = (new_h, new_w)
    old_shape = (h, w)
    return frame, old_shape, new_shape


def build_confidence_function_for_trackers(trackers, flow01):

    global frame_nb_for_plot
    global legends
    confidence_functions = dict()
    if verbose: 
        ax.imshow(latest_frame_to_show)
    for tracker_nb, tracker in enumerate(trackers):
        if tracker.enabled:
            confidence_functions[tracker_nb] = tracker.build_confidence_function(flow01, tracker_nb)

    if verbose:
        if len(latest_detections): 
            ax.scatter(latest_detections[:, 0], latest_detections[:, 1], c='r', s=40)

        ax.xaxis.tick_top()
        plt.legend(handles=legends)
        fig.canvas.draw()
        plt.show()
        while not plt.waitforbuttonpress():
            continue
        ax.cla()
        legends = []

    return confidence_functions


def track_video(detections, flows, state_variance, observation_variance, algorithm, stop_tracking_threshold, confidence_threshold):

    global latest_detections
    global frame_nb_for_plot
    global latest_frame_to_show
    init = False
    trackers = dict()

    for frame_nb in tqdm(range(len(detections))):
        detections_for_frame = detections[frame_nb]

        if verbose:
            frame_nb_for_plot = frame_nb
            latest_detections = detections_for_frame
            latest_frame_to_show = frames[frame_nb]

        if not init:
            if len(detections_for_frame):
                trackers = init_trackers(
                    detections_for_frame, frame_nb, state_variance, observation_variance, algorithm, stop_tracking_threshold)
                init = True

        else:

            flow01 = flows[frame_nb-1]
            new_trackers = []
            if verbose:
                confidence_functions_for_trackers = build_confidence_function_for_trackers(
                    trackers, flow01)
            if len(detections_for_frame):
                if not verbose:
                    confidence_functions_for_trackers = build_confidence_function_for_trackers(
                        trackers, flow01)
                assigned_trackers = - \
                    np.ones(len(detections_for_frame), dtype=int)
                assignment_confidences = -np.ones(len(detections_for_frame))

                if len(confidence_functions_for_trackers):
                    for detection_nb in range(len(detections_for_frame)):

                        tracker_scores = {tracker_nb: confidence_for_tracker(
                            detections_for_frame[detection_nb]) for tracker_nb, confidence_for_tracker in confidence_functions_for_trackers.items()}

                        tracker_ids = list(tracker_scores.keys())
                        candidate_tracker_id = tracker_ids[int(
                            np.argmax(list(tracker_scores.values())))]

                        score_for_candidate_cloud = tracker_scores[candidate_tracker_id]

                        if score_for_candidate_cloud > confidence_threshold:

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
                            Tracker(frame_nb, detection, state_variance, observation_variance, stop_tracking_threshold=stop_tracking_threshold, algorithm=algorithm))
                    else:
                        trackers[assigned_tracker].update(
                            detection, flow01, frame_nb)

            for tracker in trackers:
                tracker.update_status(flow01)

            if len(new_trackers):
                trackers.extend(new_trackers)

    results = []
    tracklets = [tracker.tracklet for tracker in trackers]
    tracklets = [tracklet for tracklet in tracklets if len(
        tracklet) > stop_tracking_threshold]

    for tracker_nb, associated_detections in enumerate(tracklets):
        for associated_detection in associated_detections:
            results.append(
                (associated_detection[0], tracker_nb, associated_detection[1][0], associated_detection[1][1]))

    results = sorted(results, key=lambda x: x[0])

    return results


def detection_results_from_images(video, annotations, data_dir, detector, flow, downsampling_factor=4):

    global frames 
    frames = []

    images_for_video = [image for image in annotations['images']
                        if image['video_id'] == video['id']]
    images_for_video = sorted(
        images_for_video, key=lambda image: image['frame_id'])

    filenames = [os.path.join(data_dir, image['file_name'])
                 for image in images_for_video]


    detections, flows = [], []
    frame_nb = 0
    frame0, old_shape, new_shape = read_and_resize(filenames[frame_nb])
    downsampled_shape = (
        new_shape[1] // downsampling_factor, new_shape[0] // downsampling_factor)
    if verbose:
        frames.append(cv2.cvtColor(cv2.resize(
            frame0, downsampled_shape), cv2.COLOR_BGR2RGB))
    detections_for_frame = detector(frame0)
    if len(detections_for_frame): detections.append(detections_for_frame)
    else: detections.append(np.array([]))

    for frame_nb in tqdm(range(1, len(filenames))):

        frame1, _, _ = read_and_resize(filenames[frame_nb])
        detections_for_frame = detector(frame1)
        if len(detections_for_frame): detections.append(detections_for_frame)
        else: detections.append(np.array([]))

        flows.append(flow(frame0, frame1))
        if verbose:
            frames.append(cv2.cvtColor(cv2.resize(
                frame1, downsampled_shape), cv2.COLOR_BGR2RGB))
        frame0 = frame1.copy()

    return detections, flows, old_shape, new_shape



def external_detection_results(video, annotations, data_dir, external_detections_dir, flow, downsampling_factor=4):

    images_for_video = [image for image in annotations['images']
                        if image['video_id'] == video['id']]
    images_for_video = sorted(
        images_for_video, key=lambda image: image['frame_id'])

    filenames = [os.path.join(data_dir, image['file_name'])
                 for image in images_for_video]

    detections, flows = [], []

    detections_filename = os.path.join(
        external_detections_dir, video['file_name']+'.txt')
    with open(detections_filename, 'r') as f:
        detections_read = [detection.split(',') for detection in f.readlines()]
    detections_from_file = defaultdict(list)
    for detection in detections_read:
        detections_from_file[int(detection[0])].append(
            [float(detection[2]), float(detection[3])])

    detections_from_file = {k: np.array(v)
                            for k, v in detections_from_file.items()}

    detections, flows = [], []
    frame_nb = 0
    frame0, old_shape, new_shape = read_and_resize(filenames[frame_nb])
    downsampled_shape = (
        new_shape[1] // downsampling_factor, new_shape[0] // downsampling_factor)
    if verbose:
        frames.append(cv2.cvtColor(cv2.resize(
            frame0, downsampled_shape), cv2.COLOR_BGR2RGB))
    if frame_nb+1 in detections_from_file.keys():
        detections.append(detections_from_file[frame_nb+1])
    else:
        detections.append(np.array([]))

    for frame_nb in range(1, len(filenames)):
        if frame_nb+1 in detections_from_file.keys():
            detections.append(detections_from_file[frame_nb+1])
        else:
            detections.append(np.array([]))

        frame1, _, _ = read_and_resize(filenames[frame_nb])
        flows.append(flow(frame0, frame1))
        if verbose:
            frames.append(cv2.cvtColor(cv2.resize(
                frame1, downsampled_shape), cv2.COLOR_BGR2RGB))
        frame0 = frame1.copy()

    return detections, flows, old_shape, new_shape


def main(args):

    def flow(frame0, frame1): return compute_flow(
        frame0, frame1, args.downsampling_factor)

    if args.detections_from_images:
        base_model = load_base(args.base_weights)
        if not args.base_only:
            extension_model = load_extension(args.extension_weights, 32)

            def detector(frame): return detect_base_extension(frame, threshold=args.detection_threshold,
                                                              base_model=base_model, extension_model=extension_model)
        else:
            def detector(frame): return detect_base(frame, threshold=args.detection_threshold,
                                                    base_model=base_model)

    state_variance = np.load(os.path.join(args.data_dir, 'state_variance.npy'))
    observation_variance = np.load(os.path.join(
        args.data_dir, 'observation_variance.npy'))


    with open(args.annotation_file, 'rb') as f:
        annotations = json.load(f)

    for video in annotations['videos']:

        # video = [video_annotation for video_annotation in annotations['videos']
        #          if video_annotation['file_name'] == 'leloing__5'][0]  # debug

        output_filename = os.path.join(
            args.output_dir, video['file_name']+'.txt')
        output_file = open(output_filename, 'w')

        if args.detections_from_images:
            detections, flows, old_shape, new_shape = detection_results_from_images(
                video, annotations, args.data_dir, detector, flow)
        else:
            detections, flows, old_shape, new_shape = external_detection_results(
                video, annotations, args.data_dir, args.external_detections_dir, flow)

        ratio_y = old_shape[0] / (new_shape[0] // args.downsampling_factor)
        ratio_x = old_shape[1] / (new_shape[1] // args.downsampling_factor)

        if not args.detections_from_images:
            detections_resized = []
            for detection in detections:
                if len(detection):
                    detections_resized.append(
                        np.array([1/ratio_x, 1/ratio_y])*detection)
                else:
                    detections_resized.append(detection)
            detections = detections_resized

        results = track_video(
            detections, flows, state_variance, observation_variance, algorithm=args.algorithm, stop_tracking_threshold=args.stop_tracking_threshold, confidence_threshold=args.confidence_threshold)

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
    parser.add_argument('--detections_from_images', action='store_true')
    parser.add_argument('--external_detections_dir', type=str)
    parser.add_argument('--base_only', action='store_true')
    parser.add_argument('--algorithm', type=str, default='Kalman')
    args = parser.parse_args()

    main(args)
