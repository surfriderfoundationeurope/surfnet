import cv2
import json
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

class GaussianMixture(object):
    def __init__(self, means, variance, weights):
        cov = variance*np.eye(2)
        self.components = [multivariate_normal(mean=mean,cov=cov) for mean in means]
        self.weights = weights

    def pdf(self, x):
        result = 0
        for weight, component in zip(self.weights, self.components): 
            result += weight*component.pdf(x)
        return result 

    def cdf(self,x):
        result = 0
        for weight, component in zip(self.weights, self.components): 
            result += weight*component.cdf(x)
        return result 

class StateSpaceModel(object):
    def __init__(self, state_transition_variance, state_observation_variance):
        self.state_transition_variance = state_transition_variance
        self.state_observation_variance = state_observation_variance

    def state_transition(self, current_state, flow):
        mean = current_state + flow[round(current_state[0]),round(current_state[1]),:][::-1]
        cov = self.state_transition_variance*np.eye(2)
        return multivariate_normal(mean, cov)

    def state_observation(self, current_state):
        mean = current_state
        cov = self.state_observation_variance*np.eye(2)
        return multivariate_normal(mean, cov)

class SMC(object):

    def __init__(self, X0, SSM, n_particles=10):

        self.n_particles = n_particles
        self.particles, self.weights = self.init_particles(X0)
        self.SSM = SSM 
        self.updated = False 
        self.countdown = 0
        self.is_on = True


    def init_particles(self, X0):

        particles = np.array(self.n_particles*[X0])
        weights = np.ones(self.n_particles)/self.n_particles

        return particles, weights 

    def state_transition(self, flow):
        new_particles = []
        for particle in self.particles: 
            new_particle = self.SSM.state_transition(particle, flow).rvs(1)
            new_particles.append(new_particle)
        self.particles = np.array(new_particles)

    def importance_reweighting(self, observation):
        weights = []
        for particle in self.particles:
            weights.append(self.SSM.state_observation(particle).pdf(observation))
        self.weights = np.array(weights)/sum(weights)

    def resample(self):
        resampling_indices = np.random.choice(a=self.n_particles, p=self.weights, size = self.n_particles)
        self.particles = self.particles[resampling_indices]
        self.weights = np.ones(self.n_particles)/self.n_particles

    def update(self, observation, flow):
        self.resample()
        self.state_transition(flow)
        self.importance_reweighting(observation)
        self.updated = True

    def update_countdown(self):
        if not self.updated: 
            self.countdown+=1
        else:
            self.countdown=0
        if self.countdown > 5: 
            self.is_on = False
    
def init_trackers(detections, SSM):
    current_trackers = []

    for detection in detections:
        particle_filter_for_detection = SMC(detection, SSM)
        current_trackers.append(particle_filter_for_detection)

    return current_trackers

def compute_flow(frame0, frame1, downsampling_factor):
    h, w = frame0.shape[:-1]

    new_h = h // downsampling_factor
    new_w = w // downsampling_factor

    frame0 = cv2.resize(frame0, (new_w, new_h))
    frame1 = cv2.resize(frame1, (new_w, new_h))

    flow01 = flow_opencv_dense(frame0, frame1)
    
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
    base_model = create_base('dla_34', heads = {'hm':1,'wh':2}, head_conv=256)
    base_model = load_my_model(base_model, base_weights)
    for param in base_model.parameters():
        param.requires_grad = False 
    base_model.to('cuda')
    base_model.eval()
    return base_model

def detect(frame, threshold, base_model, extension_model):
    frame = transform_test_CenterNet()(frame).to('cuda').unsqueeze(0)
    base_result = base_model(frame)[-1]['hm']
    extension_result = torch.sigmoid(extension_model(base_result))
    detections = nms(extension_result).gt(threshold).squeeze()
    return torch.nonzero(detections).cpu().numpy()

def read_and_resize(filename):
    frame = cv2.imread(filename)
    h, w = frame.shape[:-1]
    new_h = (h | 31) + 1
    new_w = (w | 31) + 1
    frame = cv2.resize(frame, (new_w, new_h))
    new_shape = (new_h, new_w)
    old_shape = (h,w)
    return frame, old_shape, new_shape

def build_confidence_function_for_tracker(particles, weights, SSM, flow01, nb_new_particles=5):

    new_particles = []
    new_weights = []

    for particle, weight in zip(particles, weights):
        new_particles.extend(SSM.state_transition(particle, flow01).rvs(nb_new_particles))
        new_weights.extend([weight/nb_new_particles]*nb_new_particles)

    distribution = GaussianMixture(new_particles, SSM.state_observation_variance, new_weights)

    verbose = False
    if verbose: 
        shape = flow01.shape[:-1]
        y, x = np.mgrid[0:shape[0]:1, 0:shape[1]:1]
        pos = np.dstack((x, y))
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.contourf(x, y, distribution.pdf(pos))
        plt.show()
        
    range = np.array([2,2])

    confidence_for_x = lambda x: distribution.cdf(x+range) - distribution.cdf(x-range)

    return confidence_for_x 

def build_confidence_function_for_trackers(current_trackers, SSM, flow01):
    
    confidence_functions = dict()
    for tracker_nb, tracker in enumerate(current_trackers):
        if tracker.is_on:
            confidence_functions[tracker_nb] = build_confidence_function_for_tracker(tracker.particles, tracker.weights, SSM, flow01)
    return confidence_functions

def track_video(filenames, detector, SSM, flow):

    tracklet = []

    init = False 
    frame0, _ , _ = read_and_resize(filenames[0])
    detections = detector(frame0)

    if len(detections): 
        current_trackers = init_trackers(detections, SSM)
        for detection in detections:
            tracklet.append([(0,detection)])
        init = True

    for frame_nb in range(1,len(filenames)):
        frame1, _, _ = read_and_resize(filenames[frame_nb])
        detections = detector(frame1)
        
        if len(detections):
            if not init:
                    current_trackers = init_trackers(detections, SSM)
                    init = True 

            else: 
                flow01 = flow(frame0, frame1)
                confidence_functions_for_trackers = build_confidence_function_for_trackers(current_trackers, SSM, flow01)
                assigned_trackers = -np.ones(len(detections),dtype=int)
                assignment_confidences = -np.ones(len(detections),dtype=int)
                
                for detection_nb in range(len(detections)):

                    tracker_scores = {tracker_nb:confidence_for_tracker(detections[detection_nb]) for tracker_nb, confidence_for_tracker in confidence_functions_for_trackers.items()} 
                    trackers = list(tracker_scores.keys())
                    candidate_tracker_id = trackers[int(np.argmax(tracker_scores.values()))]
                    
                    score_for_candidate_cloud = tracker_scores[candidate_tracker_id]

                    if score_for_candidate_cloud > 0.2:
                        if candidate_tracker_id in assigned_trackers:
                            detection_id_of_conflict = np.argwhere(assigned_trackers == candidate_tracker_id)
                            if score_for_candidate_cloud > assignment_confidences[detection_id_of_conflict]:
                                assigned_trackers[detection_id_of_conflict] = -1
                                assignment_confidences[detection_id_of_conflict] = -1
                                assigned_trackers[detection_nb] = candidate_tracker_id
                                assignment_confidences[detection_nb] = score_for_candidate_cloud
                        else:
                            assigned_trackers[detection_nb] = candidate_tracker_id
                            assignment_confidences[detection_nb] = score_for_candidate_cloud
                
                for detection_nb in range(len(detections)):
                    detection = detections[detection_nb]
                    assigned_tracker = assigned_trackers[detection_nb]
                    if assigned_tracker == -1:
                        current_trackers.append(SMC(detection,SSM))
                        tracklet.append([(frame_nb,detection)])
                    else: 
                        current_trackers[assigned_tracker].update(detection, flow01)
                        tracklet[assigned_tracker].append((frame_nb, detection))
                
        for tracker in current_trackers:
            tracker.update_countdown()

    return tracklet

def main(args):

    base_model = load_base(args.base_weights)
    extension_model = load_extension(args.extension_weights, 32)
 
    detector = lambda frame : detect(frame, threshold=args.confidence_threshold, base_model=base_model, extension_model=extension_model)

    flow = lambda frame0, frame1 : -compute_flow(frame0, frame1, args.downsampling_factor)

    SSM = StateSpaceModel(state_transition_variance=2, state_observation_variance=2)

    with open(args.annotation_file,'rb') as f: 
        annotations = json.load(f)
    
    for video in annotations['videos']:
        output_filename = os.path.join(args.output_dir,video['file_name']+'.txt')
        output_file = open(output_filename,'w')
        images_for_video = [image for image in annotations['images'] if image['video_id']==video['id']]
        images_for_video = sorted(images_for_video, key=lambda image:image['frame_id'])
        filenames = [os.path.join(args.data_dir,image['file_name']) for image in images_for_video]
        tracklet = track_video(filenames, detector, SSM, flow)
        output_file.close()



if __name__ == '__main__': 
    import argparse 
    parser = argparse.ArgumentParser(description='Tracking')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--annotation_file', type=str)
    parser.add_argument('--threshold_unassigned_objects',type=float,default=0.4)
    parser.add_argument('--confidence_threshold',type=float, default=0.4)
    parser.add_argument('--base_weights',type=str)
    parser.add_argument('--extension_weights',type=str)
    parser.add_argument('--output_dir',type=str)
    parser.add_argument('--downsampling_factor',type=int)


    args = parser.parse_args()
    
    main(args)
