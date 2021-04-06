from synthetic_videos.flow_tools import flow_opencv_dense
import os
import cv2
import numpy as np 
from common.utils import warp_flow

class VideoOpenCV(object):
    def __init__(self, video_name, fix_res=False, downsampling_factor=4):
        self.cap = cv2.VideoCapture(video_name)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fix_res = fix_res
        self.downsampling_factor = downsampling_factor

    def read(self):
        ret, frame = self.cap.read()

        if not ret: 
            print('Unreadable frame!')
        return self.resize(frame)

    def resize(self, frame):
        if self.fix_res:
            new_h = 512
            new_w = 512
        else:
            h, w = frame.shape[:-1]
            new_h = (h | 31) + 1
            new_w = (w | 31) + 1

        new_h = new_h // self.downsampling_factor
        new_w = new_w // self.downsampling_factor

        return cv2.resize(frame, (new_w, new_h))

def verbose(frame0, frame1, flow01):

    frame0_remapped = warp_flow(frame0,flow01)

    mag, ang = cv2.cartToPolar(flow01[...,0], flow01[...,1])
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 0
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    # nvec = 20  # Number of vectors to be displayed along each image dimension
    # nl, nc = frame0.s
    # step = max(nl//nvec, nc//nvec)

    # y, x = np.mgrid[:nl:step, :nc:step]
    # u_ = u[::step, ::step]
    # v_ = v[::step, ::step]

    # ax1.imshow(norm)
    # ax1.quiver(x, y, u_, v_, color='r', units='dots',
    #         angles='xy', scale_units='xy', lw=3)

    cv2.imshow('frame0',np.concatenate([frame0, frame1, bgr, frame0_remapped], axis=0))

    cv2.waitKey(0)

video_folder = 'data/generated_videos/'
video_names = [video_name for video_name in sorted(os.listdir(video_folder)) if '.MP4' in video_name]

for video_nb, video_name in enumerate(video_names):

    print('Processing video {}'.format(video_nb))
    video = VideoOpenCV(video_folder + video_name)
    flows_video = list()
    for i in range(video.num_frames // 2):

        frame0 = video.read()
        frame1 = video.read()

        flow01 = flow_opencv_dense(frame0, frame1)
        flows_video.append(flow01)
        np.save(video_name.strip('.MP4') +'_flows',flows_video)

    flows_video = np.stack(flows_video)










        


