import numpy as np 
from collections import defaultdict 
import argparse

def main(args):
    raw_results = np.loadtxt(args.input_file, delimiter=',')
    tracklets = defaultdict(list) 

    for result in raw_results:
        track_id = int(result[1])
        frame_id = int(result[0])
        left, top, width, height = result[2:6]
        center_x = left + width/2
        center_y = top + height/2 
        tracklets[track_id].append((frame_id, center_x, center_y))



    tracklets = [tracklet for tracklet in list(tracklets.values()) if len(tracklet) > args.min_len_tracklet]
    results = []

    for tracker_nb, associated_detections in enumerate(tracklets):
        for associated_detection in associated_detections:
            results.append(
                (associated_detection[0], tracker_nb, associated_detection[1], associated_detection[2]))

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
    parser.add_argument('--min_len_tracklet',type=int)
    parser.add_argument('--output_name',type=str)
    args = parser.parse_args()
    main(args)

    