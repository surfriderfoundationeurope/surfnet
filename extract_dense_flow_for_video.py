from synthetic_videos.flow_tools import flow_opencv_dense
import os
import cv2
import numpy as np 
from common.utils import warp_flow
from PIL import Image 
import torch 
from torchvision.transforms import ToTensor
from torch.nn.functional import grid_sample

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
def warp_flow_test(image_A, image_B, flow_AB):
    # images = grid_samples()
    # images_warped = grid_sample(images, flows, mode='bilinear')
    # image = image.permute(1,2,0).numpy()[:,:,::-1]
    # res = cv2.remap(image, flow, None, cv2.INTER_LINEAR)

    flow_AB = flow_AB.permute(0,3,1,2)
    B, C, H, W = image_A.shape

    xx = torch.arange(0, W).view(1,-1).repeat(H,1)

    yy = torch.arange(0, H).view(-1,1).repeat(1,W)

    xx = xx.view(1,1,H,W).repeat(B,1,1,1)

    yy = yy.view(1,1,H,W).repeat(B,1,1,1)

    grid = torch.cat((xx,yy),1).float()

    vgrid = grid + flow_AB

    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0

    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    warped_image = torch.nn.functional.grid_sample(image_A, vgrid.permute(0,2,3,1), 'nearest')

    import matplotlib.pyplot as plt
    fig , ((ax0, ax1),(ax2,ax3)) = plt.subplots(2,2)
    ax0.imshow(image_A[0].permute(1,2,0))
    ax1.imshow(warped_image[0].permute(1,2,0))
    ax2.set_axis_off()
    ax3.imshow(image_B[0].permute(1,2,0))
    plt.show()


    # for image, flow in zip(images,flows):
    #     h, w = flow.shape[:2]
    #     flow = -flow
    #     flow[:,:,0] += np.arange(w)
    #     flow[:,:,1] += np.arange(h)[:,np.newaxis]
    # return images_warped

def verbose(frame0, frame1, flow01):
    frame0 = ToTensor()(Image.fromarray(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))).unsqueeze(0)
    frame1 = ToTensor()(Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))).unsqueeze(0)

    # flow01=-flow01

    # h, w = flow01.shape[:2]

    flow01 = torch.from_numpy(flow01).unsqueeze(0)

    flow01 = -flow01
    # flow01[:,:,0] += np.arange(w)
    # flow01[:,:,1] += np.arange(h)[:,np.newaxis]

    # flow01[:,:,0] /= w
    # flow01[:,:,1] /= h

    # flow01 = torch.from_numpy(flow01).unsqueeze(0)
    warp_flow_test(frame0, frame1, flow01)



      # mag, ang = cv2.cartToPolar(flow01[...,0], flow01[...,1])
    # hsv = np.zeros_like(frame1)
    # hsv[...,1] = 0
    # hsv[...,0] = ang*180/np.pi/2
    # hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    # bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    # nvec = 20  # Number of vectors to be displayed along each image dimension
    # nl, nc = frame0.s
    # step = max(nl//nvec, nc//nvec)

    # y, x = np.mgrid[:nl:step, :nc:step]
    # u_ = u[::step, ::step]
    # v_ = v[::step, ::step]

    # ax1.imshow(norm)
    # ax1.quiver(x, y, u_, v_, color='r', units='dots',
    #         angles='xy', scale_units='xy', lw=3)

    # cv2.imshow('frame0',np.concatenate([frame0, frame1, bgr], axis=0))

    cv2.waitKey(0)

def main(args):
    video_names = [video_name for video_name in sorted(os.listdir(args.video_folder)) if '.MP4' in video_name]

    for video_nb, video_name in enumerate(video_names):

        print('Processing video {}'.format(video_nb))
        video = VideoOpenCV(args.video_folder + video_name)
        flows_video = list()
        for frame_nb in range(video.num_frames // 2):

            frame0 = video.read()
            frame1 = video.read()

            flow01 = flow_opencv_dense(frame0, frame1)

            verbose(frame0,frame1,flow01)
            # flows_video.append(flow01)
            # np.save(video_name.strip('.MP4') + '_flows', flows_video)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extracting optical flows')

    parser.add_argument('--video-folder',default='./data/generated_videos/')
    parser.add_argument('--output-dir',default='./')
    parser.add_argument('--downsampling-factor', type=int, dest='downsampling_factor',default=4)

    args = parser.parse_args()

    main(args)






        


