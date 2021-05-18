import cv2 
from collections import defaultdict

class VideoReader:
    def __init__(self, video_filename, skip_frames):
        self.skip_frames = skip_frames 
        self.video = cv2.VideoCapture(video_filename)
        self.shape = (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = self.video.get(cv2.CAP_PROP_FPS) / (skip_frames+1)
        self.frame_nb = 0

    def read(self):
        if self.frame_nb == 0: ret ,frame = self.video.read()
        else:
            for _ in range(self.skip_frames):
                self.video.read()
            ret ,frame = self.video.read()
        self.frame_nb+=1
        return ret, frame, self.frame_nb-1

fourcc = cv2.VideoWriter_fourcc(*'mp4v')


video_filename = '/home/mathis/Documents/datasets/surfrider/videos/gopro_video/true_validation/videos/T1/good_one/T1_1080_px_converted.mp4'
results_filename = 'experiments/tracking/external_mine_544_960_half_fps_threshold_04/T1_1080_px_converted.txt'

video = VideoReader(video_filename, skip_frames=1)
writer = cv2.VideoWriter(filename='test.mp4', apiPreference=cv2.CAP_FFMPEG, fourcc=fourcc, fps=video.fps, frameSize=video.shape, params=None)

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

font = cv2.FONT_HERSHEY_SIMPLEX
ret, frame, frame_nb = video.read()

while ret: 
    detections_for_frame = results[frame_nb]
    for detection in detections_for_frame:
        # frame = cv2.circle(frame, (int(detection[1]),int(detection[2])), 5, (255, 0, 0), -1)
        cv2.putText(frame, '{}'.format(detection[0]), (int(detection[1]), int(detection[2])), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    writer.write(frame)
    ret, frame, frame_nb = video.read()

writer.release()



    
