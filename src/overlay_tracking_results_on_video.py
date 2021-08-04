import cv2 
from collections import defaultdict
from tools.video_readers import SimpleVideoReader

def main(args):

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    write=args.write
    video_filename = args.input_video
    results_filename = args.input_mot_file
    gt_filename = args.input_gt_mot_file

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
    
    if gt_filename is not None:
        with open(gt_filename, 'r') as f: 
            gt_results_raw = f.readlines()
            gt_results = defaultdict(list)
            for line in gt_results_raw:
                line = line.split(',')
                frame_nb = int(line[0]) - 1
                object_nb = int(line[1])
                center_x = float(line[2])
                center_y = float(line[3])
                gt_results[frame_nb].append((object_nb, center_x, center_y))

    video = SimpleVideoReader(video_filename, skip_frames=args.skip_frames)
    if write: 
        writer = cv2.VideoWriter(filename=args.output_name+'.mp4', 
                                    apiPreference=cv2.CAP_FFMPEG, 
                                    fourcc=fourcc, 
                                    fps=video.fps, 
                                    frameSize=video.shape, 
                                    params=None)





    font = cv2.FONT_HERSHEY_SIMPLEX
    ret, frame, frame_nb = video.read()
    while ret: 

        detections_for_frame = results[frame_nb]
        for detection in detections_for_frame:
            # if detection[0] >= 48 and detection[0] <= 50:
            frame = cv2.circle(frame, (int(detection[1]), int(detection[2])), 5, (0, 0, 255), -1)
            cv2.putText(frame, '{}'.format(detection[0]), (int(detection[1]), int(detection[2])+20), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        if gt_filename is not None: 
            gt_for_frame = gt_results[frame_nb]
            for gt in gt_for_frame:
                # frame = cv2.circle(frame, (int(gt[1]), int(gt[2])), 5, (255, 0, 0), -1)
                cv2.putText(frame, '{}'.format(gt[0]), (int(gt[1]), int(gt[2])+10), font, 2, (255, 0, 0), 2, cv2.LINE_AA)
    
        if write: writer.write(frame)
        else: 
            # if heatmaps_filename is not None: 
            cv2.imshow('tracking_results', frame)
            # cv2.imshow('heatmap', cv2.resize(heatmaps[frame_nb].cpu().numpy(),frame.shape[:-1][::-1]))
            cv2.waitKey(0)
        ret, frame, frame_nb = video.read()
    if write: writer.release()


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video',type=str)
    parser.add_argument('--input_mot_file',type=str)
    parser.add_argument('--input_gt_mot_file',type=str)
    parser.add_argument('--write',type=bool)
    parser.add_argument('--output_name',type=str,default='overlay')
    parser.add_argument('--skip_frames',type=int,default=0)
    args = parser.parse_args()
    main(args)