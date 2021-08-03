import numpy as np 
from collections import defaultdict 
import argparse
from tools.video_readers import IterableFrameReader
from tools.optical_flow import compute_flow
from pykalman import AdditiveUnscentedKalmanFilter
from numpy import ma
from tqdm import tqdm 

class UKFSmoother:
    def __init__(self, transition_variance, observation_variance, flows):
        transition_functions = [lambda x: x + flow[int(x[1]),int(x[0]),:] for flow in flows]
        self.smoother = AdditiveUnscentedKalmanFilter(transition_functions=transition_functions,
                                                 observation_functions=lambda z: np.eye(2).dot(z),
                                                 transition_covariance=np.diag(transition_variance),
                                                 observation_covariance=np.diag(observation_variance))


    def predictive_distribution(self):
        return 
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

    if args.filter_type == 'v0':

        tracks = filter_by_nb_obs(tracklets, args.min_len_tracklet)

    elif args.filter_type == 'v1':
        tracks = filter_by_mean_consecutive_length(tracklets, args.min_mean)

    elif args.filter_type == 'smoothing_v0':
        tracks = filter_from_smoothing(tracklets, args.video_filename)

    results = []
    for tracker_nb, associated_detections in enumerate(tracks):
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

def filter_by_nb_obs(tracklets, min_len_tracklet):

    return [tracklet for tracklet in tracklets if len(tracklet) > min_len_tracklet]

def filter_by_mean_consecutive_length(tracklets, min_mean):

    tracks = []

    for tracklet in tracklets: 

        consecutive_parts = [[tracklet[0]]]

        for obs in tracklet[1:]:

            previous_frame_id = consecutive_parts[-1][-1][0]

            if obs[0] == previous_frame_id + 1:
                consecutive_parts[-1].append(obs)
            else:
                consecutive_parts.append([obs])

        consecutive_parts_lengths = [len(consecutive_part) for consecutive_part in consecutive_parts]
        consecutive_parts_lengths_mean = np.mean(consecutive_parts_lengths)

        if consecutive_parts_lengths_mean > min_mean:
            tracks.append(tracklet)
            
    return tracks

def filter_from_smoothing(tracklets, video_filename):
    downsampling_factor = 4
    observation_variance = np.load('data/tracking_parameters/observation_variance.npy')
    transition_variance = np.load('data/tracking_parameters/transition_variance.npy')
    reader = IterableFrameReader(video_filename, skip_frames=0, output_shape=(960,544))
    frame0 = next(reader)
    flows = []
    for frame1 in tqdm(reader):
        flows.append(compute_flow(frame0, frame1, downsampling_factor))
        frame0 = frame1.copy()

    ratio_x = flows[0].shape[1] / 1920 
    ratio_y = flows[0].shape[0] / 1080

    for tracklet in tracklets: 
        first_frame_nb = tracklet[0][0] - 1
        last_frame_nb = tracklet[-1][0] - 1
        flows_for_tracklet = flows[first_frame_nb:last_frame_nb]
        observations = ma.empty(shape=(last_frame_nb-first_frame_nb+1,2))
        observations.mask = True
        for (frame_nb, center_x, center_y) in tracklet:
            observations[frame_nb-1] = ratio_x * center_x, ratio_y * center_y

        transition_functions = [lambda x: x + flow[int(x[1]),int(x[0]),:] for flow in flows_for_tracklet]

        smoother = AdditiveUnscentedKalmanFilter(transition_functions=transition_functions,
                                                 observation_functions=lambda z: np.eye(2).dot(z),
                                                 transition_covariance=np.diag(transition_variance),
                                                 observation_covariance=np.diag(observation_variance))

        smoothed_state_means, smoothed_state_covariances = smoother.smooth(observations)


        return 


    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',type=str)
    parser.add_argument('--min_len_tracklet',type=int)
    parser.add_argument('--output_name',type=str)
    parser.add_argument('--filter_type',type=str)
    parser.add_argument('--min_mean',type=float)
    parser.add_argument('--video_filename',type=str)
    args = parser.parse_args()
    main(args)

    