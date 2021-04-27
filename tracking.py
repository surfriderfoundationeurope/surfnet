from extract_and_save_heatmaps import save_flow
import cv2
import json
from base.centernet.models import create_model as create_base
from common.utils import load_my_model, warp_flow
from extension.models import SurfNet
import torch
from torch.nn import Module
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
        mean = current_state + flow[current_state[0],current_state[1],:][::-1]
        cov = self.state_transition_variance*np.eye(2)
        return multivariate_normal(mean, cov)

    def state_observation(self, current_state):
        mean = current_state
        cov = self.state_observation_variance*np.eye(2)
        return multivariate_normal(mean,cov)

class SMC(object):

    def __init__(self, X0, SSM, n_particles=20):

        self.n_particles = n_particles
        self.particles, self.weights = self.init_particles(X0)
        self.SSM = SSM 

    def init_particles(self, X0):

        particles = np.array(self.n_particles*[X0])
        weights = np.ones(particles.shape[0])/self.n_particles

        return particles, weights 

    def state_transition(self, flow):
        new_particles = []
        for particle in self.particles: 
            new_particle = self.SSM.state_transition(particle, flow).rvs(1)
            new_particles.append(new_particle)
        return new_particles

    def importance_reweighting(self, observation):
        weights = []
        for particle in self.particles:
            self.weights.append(self.SSM.state_observation(particle).pdf(observation))
        self.weights = np.array(weights)/sum(weights)

    def resample(self):
        return 

    def update(self, observation, flow):
        self.resample()
        self.state_transition(flow)
        self.importance_reweighting(observation)

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

def build_next_estimate_for_tracker(particles, weights, SSM, flow01, nb_new_particles=10):
    new_particles = []
    new_weights = []

    for particle, weight in zip(particles, weights):
        new_particles.extend(SSM.state_transition(particle, flow01).rvs(nb_new_particles))
        new_weights.extend([weight]*nb_new_particles)

    distribution = GaussianMixture(new_particles, SSM.state_observation_variance, weights)

    range = np.array([5,5])

    confidence_for_x = lambda x: distribution.cdf(x+range) - distribution.cdf(x-range)

    return confidence_for_x 

def build_next_estimate_for_trackers(current_trackers, SSM, flow01):
    
    pdfs = []
    for tracker in current_trackers:
        pdfs.append(build_next_estimate_for_tracker(tracker.particles, tracker.weights, SSM, flow01))

    return pdfs

def track_video(filenames, output_file, detector, SSM, flow):

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
                next_estimate_for_trackers = build_next_estimate_for_trackers(current_trackers, SSM, flow01)
                assigned_trackers_for_detections = []
                for detection in detections:
                    scores_detection = np.array([next_estimate_for_tracker(detection) for next_estimate_for_tracker in next_estimate_for_trackers])
                    candidate_cloud_id = np.argmax(scores_detection)
                    if scores_detection[candidate_cloud_id] > 0.2:
                        assigned_trackers_for_detections.append(candidate_cloud_id)
                    else: 
                        assigned_trackers_for_detections.append(-1)
                


            
    return 0











    
    # for filename in filenames[1:]: 

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
        track_video(filenames, output_file, detector, SSM, flow)
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
