import numpy as np
import argparse
from scipy.signal import convolve
from plasticorigins.tracking.utils import write_tracking_results_to_file, read_tracking_results
from collections import defaultdict

import json


def filter_tracks(tracklets, kappa, tau):
    """ filters the tracks depending on params
    kappa: size of the moving average window
    tau: minimum length of tracklet
    returns raw filtered tracks
    """
    if not kappa == 1:
        tracks = filter_by_nb_consecutive_obs(tracklets, kappa, tau)
    else: tracks = tracklets
    results = []
    for tracker_nb, dets in enumerate(tracks):
        for det in dets:
            # Confidence: 1.0, class: 0 (fragment)
            results.append((det[0], tracker_nb, det[1], det[2], 1.0, 0))

    results = sorted(results, key=lambda x: x[0])
    return results


def postprocess_for_api(results, class_dict=defaultdict(lambda: "fragment")):
    """ Converts tracking results into json object for API
    """
    result_list = []
    id_list = {}

    for res in results:
        frame_number = res[0]
        box = [round(res[2],1), round(res[3],1), round(res[2],1), round(res[3],1)]
        id = res[1]
        conf = round(res[4],2)
        classname = class_dict[res[5]]
        # if the id is not already is the results, add a new jsonline
        if id not in id_list:
            id_list[id] = len(result_list)
            result_list.append({"label":classname, "id": id, "frame_to_box": {str(frame_number): box}})

            # otherwise, retrieve the jsonline and append the box
        else:
            result_list[id_list[id]]["frame_to_box"][str(frame_number)] = box

    return {"detected_trash": result_list}


def count_objects(input_json, class_dict):
    results = {v:0 for v in class_dict.values()}
    total = 0
    for trash in input_json["detected_trash"]:
        results[trash["label"]] += 1
        total += 1

    return {k+f": {str(v)}":v/total for k,v in results.items()}

def write(results, output_name):
    """ Writes the results in two files:
    - tracking in a Mathis format xxx_track.txt (frame, id, box_x, box_y, ...)
    - the number of detected objects in a separate file xxx_count.txt
    """
    output_tracks_filename = output_name.split('.')[0]+'_tracks.txt'
    write_tracking_results_to_file(results, ratio_x=1, ratio_y=1,
                                   output_filename=output_tracks_filename)

    with open(output_name.split('.')[0]+'_count.txt','w') as out_file:
        if len(results):
            out_file.write(f'{max(result[1]+1 for result in results)}')
        else:
            out_file.write('0')


def threshold(tracklets, tau):
    return [tracklet for tracklet in tracklets if len(tracklet) > tau]


def compute_moving_average(tracklet, kappa):
    if len(tracklet)==0 or len(tracklet[0])==0:
        return tracklet
    pad = (kappa-1)//2
    observation_points = np.zeros(tracklet[-1][0] - tracklet[0][0] + 1)
    first_frame_id = tracklet[0][0] - 1
    for observation in tracklet:
        frame_id = observation[0] - 1
        observation_points[frame_id - first_frame_id] = 1
    density_fill = convolve(observation_points, np.ones(kappa)/kappa, mode='same')
    if pad>0 and len(observation_points) >= kappa:
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

    tracklets = read_tracking_results(args.input_file)
    filtered_results = filter_tracks(tracklets, args.kappa, args.tau)
    if args.output_type == "api":
        output = postprocess_for_api(filtered_results)
        with open(args.output_name, 'w') as f:
            json.dump(output, f)
    else:
        write(filtered_results, args.output_name)
