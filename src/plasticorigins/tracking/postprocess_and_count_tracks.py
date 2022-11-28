"""The ``postprocess_and_count_tracks`` submodule provides several functions to post process and count tracks.

The module allows the user to :

- filter tracks
- process object class and confidences
- count objects on the frames

This submodule contains the following functions :

- ``compute_moving_average(tracklet:List[List], kappa:int)`` : Computing the moving average of the tracks depending on parameter ``kappa``.
- ``count_objects(input_json:Dict, class_dict:Dict)`` : Counting trashs from the ``input_json`` object.
- ``filter_by_nb_consecutive_obs(tracklets:List[List], kappa:int, tau:int)`` : Filters the tracks depending on parameters ``kappa`` and ``tau`` by consecutive observations.
- ``filter_tracks(tracklets:List[List], kappa:int, tau:int)`` : Filters the tracks depending on parameters ``kappa`` and ``tau``.
- ``postprocess_for_api(results:List[Tuple], class_dict:Dict=defaultdict(lambda: "fragment"))`` : Converts tracking results into json object for API.
- ``process_class_and_confidences(class_confs:List[Tuple[int,float]])`` : Finds the majority and most confident class from list `class_confs`.
- ``threshold(tracklets:List[List], tau:int)`` : Filters the tracks depending on parameter ``tau``.

"""

import argparse
import json
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np
from numpy import array
from scipy.signal import convolve

from plasticorigins.tracking.utils import (
    read_tracking_results,
    write_tracking_results_to_file,
)


def filter_by_nb_consecutive_obs(tracklets: List[List], kappa: int, tau: int) -> List:

    """Filters the tracks depending on parameters ``kappa`` and ``tau`` by consecutive observations.

    Args:
        tracklets (List[List]): list of tracks to filter
        kappa (int): size of the moving average window
        tau (int): minimum length of tracklet

    Returns:
        results (List[Tuple]) : raw filtered tracks of minimum length ``tau``
    """

    new_tracklets = []

    for tracklet in tracklets:
        new_tracklet = []
        density_fill = compute_moving_average(tracklet, kappa=kappa)
        for (observation, density_fill_value) in zip(tracklet, density_fill):
            if density_fill_value > 0.6:
                new_tracklet.append(observation)
        new_tracklets.append(new_tracklet)

    return threshold(new_tracklets, tau)


def filter_tracks(tracklets: List[List], kappa: int, tau: int) -> List[Tuple]:

    """Filters the tracks depending on parameters ``kappa`` and ``tau``.

    Args:
        tracklets (List[List]): list of tracks to filter
        kappa (int): size of the moving average window
        tau (int): minimum length of tracklet

    Returns:
        results (List[Tuple]) : raw filtered tracks
    """

    if not kappa == 1:
        tracks = filter_by_nb_consecutive_obs(tracklets, kappa, tau)
    else:
        tracks = tracklets

    results = []

    for tracker_nb, dets in enumerate(tracks):
        for det in dets:
            results.append((det[0], tracker_nb, det[1], det[2], det[3], det[4]))

    results = sorted(results, key=lambda x: x[0])

    return results


def process_class_and_confidences(
    class_confs: List[Tuple[int, float]]
) -> Tuple[int, float]:

    """Finds the majority and most confident class from list `class_confs`.

    Args:
        class_confs (List[Tuple[int,float]]): the list of class ids and confidences such as ``[(class_id, confidence), ...]``

    Returns:
        best class (Tuple[int,float]) : The best class and its associated confidence
    """

    d = defaultdict(lambda: (0, 0.0))

    for (cls, conf) in class_confs:
        d[cls] = (d[cls][0] + 1, d[cls][1] + conf)

    best_class = sorted(d.items(), key=lambda v: v[1][0] + v[1][1])[-1]

    return best_class[0], round(best_class[1][1] / best_class[1][0], 2)


def postprocess_for_api(
    results: List[Tuple], class_dict: Dict = defaultdict(lambda: "fragment")
) -> Dict:

    """Converts tracking results into json object for API.

    Args:
        results (List[Tuple]) : raw filtered tracks
        class_dict (Dict): the dictionnary of object classes

    Returns:
        The detected trashs dictionnary (json object format) ``{"detected_trash" : result_list}`` with labels and average confidences in `result_list`
    """

    result_list = []
    id_list = {}

    for res in results:
        frame_number = res[0]
        box = [
            round(res[2], 1),
            round(res[3], 1),
            round(res[2], 1),
            round(res[3], 1),
        ]
        id = res[1]
        conf = round(res[4], 2)
        classname = class_dict[res[5]]
        # if the id is not already is the results, add a new jsonline
        if id not in id_list:
            id_list[id] = len(result_list)
            result_list.append(
                {
                    "label": classname,
                    "id": id,
                    "frame_to_box": {str(frame_number): box},
                    "frame_to_class_conf": {str(frame_number): (res[5], conf)},
                }
            )
        # otherwise, retrieve the jsonline and append the box
        else:
            result_list[id_list[id]]["frame_to_box"][str(frame_number)] = box
            result_list[id_list[id]]["frame_to_class_conf"][str(frame_number)] = (
                res[5],
                conf,
            )

    # Finally, collapse the confidence and class
    for res in result_list:
        classid, avg_conf = process_class_and_confidences(
            res.pop("frame_to_class_conf").values()
        )
        res["avg_conf"] = avg_conf
        # update the label
        res["label"] = class_dict[classid]

    return {"detected_trash": result_list}


def count_objects(input_json: Dict, class_dict: Dict) -> Dict:

    """Counting trashs from the ``input_json`` object.

    Args:
        input_json (Dict) : the detected trashs dictionnary ``{"detected_trash" : result_list}``
        class_dict (Dict): the dictionnary of object classes

    Returns:
        The dictionnary of trash proportions.
    """

    results = {v: 0 for v in class_dict.values()}
    total = 0

    for trash in input_json["detected_trash"]:
        results[trash["label"]] += 1
        total += 1

    if total == 0:
        total = 1

    return {k + f": {str(v)}": v / total for k, v in results.items()}


def write(results: List[Tuple], output_name: str):

    """Writes the results in two files:
    - tracking in a ``xxx_track.txt`` format  ``(frame, id, box_x, box_y, ...)``
    - the number of detected objects in a separate file ``xxx_count.txt``

    Args:
        results (List[Tuple]) : raw filtered tracks
        output_name (str): the name of the output file for tracking and counting
    """

    output_tracks_filename = output_name.split(".")[0] + "_tracks.txt"
    write_tracking_results_to_file(
        results, lambda x, y: (x, y), output_filename=output_tracks_filename
    )

    with open(output_name.split(".")[0] + "_count.txt", "w") as out_file:
        if len(results):
            out_file.write(f"{max(result[1]+1 for result in results)}")
        else:
            out_file.write("0")


def threshold(tracklets: List[List], tau: int) -> List:

    """Filters the tracks depending on parameter ``tau``.

    Args:
        tracklets (List[List]): list of tracks to filter
        tau (int): minimum length of tracklet

    Returns:
        results (List) : raw filtered tracks
    """

    return [tracklet for tracklet in tracklets if len(tracklet) > tau]


def compute_moving_average(tracklet: List[List], kappa: int) -> array:

    """Computing the moving average of the tracks depending on parameter ``kappa``.

    Args:
        tracklet (List[List]): list of tracks to filter
        kappa (int): size of the moving average window

    Returns:
        density_fill (array) : moving average resulting from convolution between observation points and ``kappa``
    """

    if len(tracklet) == 0 or len(tracklet[0]) == 0:
        return tracklet

    pad = (kappa - 1) // 2
    observation_points = np.zeros(tracklet[-1][0] - tracklet[0][0] + 1)
    first_frame_id = tracklet[0][0] - 1

    for observation in tracklet:
        frame_id = observation[0] - 1
        observation_points[frame_id - first_frame_id] = 1

    density_fill = convolve(observation_points, np.ones(kappa) / kappa, mode="same")

    if pad > 0 and len(observation_points) >= kappa:
        density_fill[:pad] = density_fill[pad : 2 * pad]
        density_fill[-pad:] = density_fill[-2 * pad : -pad]

    density_fill = observation_points * density_fill

    return density_fill[density_fill > 0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--kappa", type=int)
    parser.add_argument("--tau", type=int)
    parser.add_argument("--output_type", type=str, default="api")
    args = parser.parse_args()

    tracklets = read_tracking_results(args.input_file)
    filtered_results = filter_tracks(tracklets, args.kappa, args.tau)
    if args.output_type == "api":
        output = postprocess_for_api(filtered_results)
        with open(args.output_name, "w") as f:
            json.dump(output, f)
    else:
        write(filtered_results, args.output_name)
