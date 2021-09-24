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

    if not args.kappa == 1:
        tracks = filter_by_nb_consecutive_obs(tracklets, args.kappa, args.tau)
    else: tracks = tracklets
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
            
def threshold(tracklets, tau):

    return [tracklet for tracklet in tracklets if len(tracklet) > tau]

def compute_moving_average(tracklet, kappa):

    pad = (kappa-1)//2
    observation_points = np.zeros(tracklet[-1][0] - tracklet[0][0] + 1)
    first_frame_id = tracklet[0][0] - 1
    for observation in tracklet:
        frame_id = observation[0] - 1
        observation_points[frame_id - first_frame_id] = 1
    density_fill = convolve(observation_points, np.ones(kappa)/kappa, mode='same')
    if len(observation_points) >= kappa:
        density_fill[:pad] = density_fill[pad:2*pad]
        density_fill[-pad:] = density_fill[-2*pad:-pad]
    density_fill = observation_points * density_fill

    return  density_fill[density_fill > 0]

def filter_by_nb_consecutive_obs(tracklets, kappa, tau):

    new_tracklets = []

    for tracklet in tracklets: 
        new_tracklet = []
        density_fill = compute_moving_average(tracklet, kappa=kappa)
        for (observation, density_fill_value) in zip(tracklet, density_fill):
            if density_fill_value > 0.6:
                new_tracklet.append(observation)
        new_tracklets.append(new_tracklet)

    return threshold(new_tracklets, tau)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',type=str)
    parser.add_argument('--output_name',type=str)
    parser.add_argument('--kappa',type=int)
    parser.add_argument('--tau',type=int)
    args = parser.parse_args()
    main(args)

    