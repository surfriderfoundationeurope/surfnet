import cv2 
from collections import defaultdict
from common.opencv_tools import SimpleVideoReader

def main(args):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    write=args.write
    video_filename = args.input_video
    results_filename = args.input_mot_file
    # heatmaps_filename = 'data/detector_results/real_val/mine/no_early_stopping_threshold_05/T1_1080_px_converted_heatmaps.pickle'
    heatmaps = None
    video = SimpleVideoReader(video_filename, skip_frames=0)
    if write: 
        writer = cv2.VideoWriter(filename=args.output_name+'.mp4', 
                                    apiPreference=cv2.CAP_FFMPEG, 
                                    fourcc=fourcc, 
                                    fps=video.fps, 
                                    frameSize=video.shape, 
                                    params=None)

    with open(results_filename, 'r') as f: 
        results_raw = f.readlines()
        results = defaultdict(list)
        for line in results_raw:
            line = line.split(',')
            frame_nb = int(line[0]) - 1
            object_nb = int(line[1])
            center_x = float(line[2])
            center_y = float(line[3])
            results[frame_nb].append((object_nb, center_x, center_y))

    # if heatmaps_filename is not None: 
    #     with open(heatmaps_filename,'rb') as f:
    #         heatmaps = pickle.load(f)

    font = cv2.FONT_HERSHEY_SIMPLEX
    ret, frame, frame_nb = video.read()
    while ret: 
        detections_for_frame = results[frame_nb]
        for detection in detections_for_frame:
            frame = cv2.circle(frame, (int(detection[1]), int(detection[2])), 5, (255, 0, 0), -1)
            cv2.putText(frame, '{}'.format(detection[0]), (int(detection[1]), int(detection[2])), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        if write: writer.write(frame)
        else: 
            # if heatmaps_filename is not None: 
            cv2.imshow('tracking_results',frame)
            # cv2.imshow('heatmap', cv2.resize(heatmaps[frame_nb].cpu().numpy(),frame.shape[:-1][::-1]))
            cv2.waitKey(0)
        ret, frame, frame_nb = video.read()
    if write: writer.release()


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video',type=str)
    parser.add_argument('--input_mot_file',type=str)
    parser.add_argument('--write',type=bool)
    parser.add_argument('--output_name',type=str,default='overlay')
    args = parser.parse_args()
    main(args)