from pykalman import AdditiveUnscentedKalmanFilter
from scipy.stats import multivariate_normal
from tracking.utils import confidence_from_multivariate_distribution, FramesWithInfo
from tools.optical_flow import compute_flow
from collections import defaultdict
import pickle
import numpy as np 
from numpy import ma
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import argparse 

class UKFSmoother:

    def __init__(self, transition_variance, observation_variance, flows, delta):

        self.observation_covariance = np.diag(observation_variance)
        transition_functions = [lambda x: x + flow[int(x[1]),int(x[0]),:] for flow in flows]
        self.smoother = AdditiveUnscentedKalmanFilter(transition_functions=transition_functions,
                                                 observation_functions=lambda z: np.eye(2).dot(z),
                                                 transition_covariance=np.diag(transition_variance),
                                                 observation_covariance=self.observation_covariance)
        self.delta = delta

    def predictive_probabilities(self, observations):

        smoothed_state_means, smoothed_state_covariances = self.smoother.smooth(observations)

        valid_indices = ~observations.mask[:,0]
        observations = observations[valid_indices]
        smoothed_state_means = smoothed_state_means[valid_indices]
        smoothed_state_covariances = smoothed_state_covariances[valid_indices]

        predictive_probabilities = []
        for observation, smoothed_state_mean, smoothed_state_covariance in zip(observations, smoothed_state_means, smoothed_state_covariances):

            predictive_distribution = multivariate_normal(smoothed_state_mean, smoothed_state_covariance+self.observation_covariance)

            predictive_probabilities.append(confidence_from_multivariate_distribution(observation, predictive_distribution, delta=self.delta))
        
        return predictive_probabilities


def main(args):
    raw_results = np.loadtxt(args.input_file, delimiter=',')
    if raw_results.ndim == 1: raw_results = np.expand_dims(raw_results,axis=0)
    tracklets = defaultdict(list) 
    for result in raw_results:
        track_id = int(result[1])
        frame_id = int(result[0])
        left, top, width, height = result[2:6]
        center_x = left + width/2
        center_y = top + height/2 
        tracklets[track_id].append((frame_id, center_x, center_y))

    tracklets = list(tracklets.values())

    with open(args.frames_file,'rb') as f:
        frames = pickle.load(f)

    reader = FramesWithInfo(frames)
    observation_variance = np.load('data/tracking_parameters/observation_variance.npy')
    transition_variance = np.load('data/tracking_parameters/transition_variance.npy')

    delta = 0.05*euclidean(reader.output_shape, np.array([0,0]))

    print('-- Computing flows...')
    frame0 = next(reader)
    flows = []
    for frame1 in tqdm(reader):
        flows.append(compute_flow(frame0, frame1, args.downsampling_factor))
        frame0 = frame1.copy()
    
    ratio_x = flows[0].shape[1] / 1920 
    ratio_y = flows[0].shape[0] / 1080

    smoothed_tracklets = []
    print('-- Smoothing tracklets')
    for tracklet in tqdm(tracklets): 
        if len(tracklet) > 1:
            first_frame_nb = tracklet[0][0] - 1
            last_frame_nb = tracklet[-1][0] - 1
            flows_for_tracklet = flows[first_frame_nb:last_frame_nb]
            observations = ma.empty(shape=(last_frame_nb-first_frame_nb+1,2))
            observations.mask = True
            for (frame_nb, center_x, center_y) in tracklet:
                observations[frame_nb-(first_frame_nb+1)] = ratio_x * center_x, ratio_y * center_y

            smoother = UKFSmoother(transition_variance, observation_variance, flows_for_tracklet, delta)
            smoothed_predictive_probabilities = smoother.predictive_probabilities(observations)

            smoothed_tracklet = [observation for observation, proba in zip(tracklet, smoothed_predictive_probabilities) if proba > args.confidence_threshold]

            smoothed_tracklets.append(smoothed_tracklet)

        else: 
            smoothed_tracklets.append(tracklet)


    results = []
    for tracker_nb, associated_detections in enumerate(smoothed_tracklets):
        for associated_detection in associated_detections:
            results.append((associated_detection[0], tracker_nb, associated_detection[1], associated_detection[2]))

    results = sorted(results, key=lambda x: x[0])

    with open(args.output_name.split('.')[0]+'.txt','w') as out_file:

        for result in results:
            out_file.write('{},{},{},{},{},{},{},{},{},{}\n'.format(result[0],
                                                                    result[1]+1,
                                                                    result[2],
                                                                    result[3],
                                                                    -1,
                                                                    -1,
                                                                    1,
                                                                    -1,
                                                                    -1,
                                                                    -1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',type=str)
    parser.add_argument('--output_name',type=str)
    parser.add_argument('--frames_file',type=str)

    parser.add_argument('--confidence_threshold',type=float)
    parser.add_argument('--downsampling_factor',type=int)
    args = parser.parse_args()
    main(args)

