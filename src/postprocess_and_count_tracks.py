import numpy as np
from collections import defaultdict
import argparse
from scipy.signal import convolve
import json

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
    results = filter_tracks(tracklets, args.kappa, args.tau)
    if args.output_type == "api":
        output = postprocess_for_api(results)
        with open(args.output_file, 'w') as f:
            json.dump(output, f)
    else:
        write(results, args)


def filter_tracks(tracklets, kappa, tau):
    if not kappa == 1:
        tracks = filter_by_nb_consecutive_obs(tracklets, kappa, tau)
    else: tracks = tracklets
    results = []
    for tracker_nb, associated_detections in enumerate(tracks):
        for associated_detection in associated_detections:
            results.append((associated_detection[0], tracker_nb, associated_detection[1], associated_detection[2]))

    results = sorted(results, key=lambda x: x[0])
    return results


def postprocess_for_api(results):
    result_list = []
    id_list = {}

    for res in results:
        frame_number = res[0]
        box = [res[2], res[3], res[2], res[3]]
        id = res[1]
        # if the id is not already is the results, add a new jsonline
        if id not in id_list:
            id_list[id] = len(result_list)
            result_list.append({"label":"fragments",
                                "id": id,
                                "frame_to_box": {str(frame_number): box}})
        # otherwise, retrieve the jsonline and append the box
        else:
            result_list[id_list[id]]["frame_to_box"][str(frame_number)] = box
    return {"detected_trash": result_list}


def write(results, args):
    with open(args.output_name.split('.')[0]+'_tracks.txt','w') as out_file:
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

    with open(args.output_name.split('.')[0]+'_count.txt','w') as out_file:
        if len(results):
            out_file.write(f'{max(result[1]+1 for result in results)}')
        else:
            out_file.write('0')


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
    parser.add_argument('--output_type',type=str,default="api")
    args = parser.parse_args()
    main(args)
