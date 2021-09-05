import numpy as np 
from collections import defaultdict 
import argparse
from torch import max_pool1d
import torch
from scipy.signal import convolve
# import scipy.ndimage.filters as ndif


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

    elif args.filter_type.startswith('v2'):
        kernel_size = int(args.filter_type.split('_')[1])
        tracks = filter_by_nb_consecutive_obs(tracklets, args.min_len_tracklet, kernel_size)

    else:
        raise NotImplementedError
        
    results = []
    for tracker_nb, associated_detections in enumerate(tracks):
        for associated_detection in associated_detections:
            results.append((associated_detection[0], tracker_nb, associated_detection[1], associated_detection[2]))

    results = sorted(results, key=lambda x: x[0])

    with open(args.output_name.split('.')[0]+'.txt','w') as out_file:
        if len(results):
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

def compute_tracklet_density_fill(tracklet, kernel_size):

    pad = (kernel_size-1)//2
    observation_points = np.zeros(tracklet[-1][0] - tracklet[0][0] + 1)
    first_frame_id = tracklet[0][0] - 1
    for observation in tracklet:
        frame_id = observation[0] - 1
        observation_points[frame_id - first_frame_id] = 1
    density_fill = convolve(observation_points, np.ones(kernel_size)/kernel_size, mode='same')
    if len(observation_points) >= kernel_size:
        density_fill[:pad] = density_fill[pad:2*pad]
        density_fill[-pad:] = density_fill[-2*pad:-pad]
    density_fill = observation_points * density_fill

    return  density_fill[density_fill > 0]

def filter_by_nb_consecutive_obs(tracklets, min_len_tracklet, kernel_size):

    new_tracklets = []

    for tracklet in tracklets: 
        new_tracklet = []
        density_fill = compute_tracklet_density_fill(tracklet, kernel_size=kernel_size)
        for (observation, density_fill_value) in zip(tracklet, density_fill):
            if density_fill_value > 0.6:
                new_tracklet.append(observation)
        new_tracklets.append(new_tracklet)

    return filter_by_nb_obs(new_tracklets, min_len_tracklet)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',type=str)
    parser.add_argument('--min_len_tracklet',type=int)
    parser.add_argument('--output_name',type=str)
    parser.add_argument('--filter_type',type=str)
    parser.add_argument('--min_mean',type=float)
    args = parser.parse_args()
    main(args)

    