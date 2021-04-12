from extract_and_save_heatmaps import save_flow
import cv2
import json
from base.centernet.models import create_model as create_base
from common.utils import load_my_model
from extension.models import SurfNet
import torch
from torch.nn import Module
import matplotlib.pyplot as plt 
from common.utils import transform_test_CenterNet, nms
from synthetic_videos.flow_tools import flow_opencv_dense
import numpy as np


class VideoOpenCV(object):

    def __init__(self, video_name, fix_res=False, downsampling_factor=4):
        self.cap = cv2.VideoCapture(video_name)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fix_res = fix_res
        self.downsampling_factor = downsampling_factor
        self.annotation = json.load(open(video_name.replace('.MP4','.json') ,'r'))
        self.index = 0 

    def read(self):
        ret, frame = self.cap.read()

        if not ret: 
            print('Unreadable frame!')
        return frame

    def resize_frame(self, frame):
        if self.fix_res:
            new_h = 512
            new_w = 512
        else:
            h, w = frame.shape[:-1]
            new_h = (h | 31) + 1
            new_w = (w | 31) + 1

        # new_h = new_h // self.downsampling_factor
        # new_w = new_w // self.downsampling_factor

        return cv2.resize(frame, (new_w, new_h))

    def resize_annotation(self, annotation, old_frame_shape, new_frame_shape):

        old_shape_x, old_shape_y = old_frame_shape
        new_shape_x, new_shape_y = new_frame_shape

        ratio_x = new_shape_x / old_shape_x
        ratio_y = new_shape_y / old_shape_y

        for object_nb in annotation.keys():

            [top_left_x, top_left_y, width, height] = annotation[object_nb]['bbox']
            [center_x, center_y] = annotation[object_nb]['center']

            annotation[object_nb]['bbox'] = [int(top_left_x * ratio_x), int(top_left_y * ratio_y), int(width * ratio_x), int(height * ratio_y)]
            annotation[object_nb]['center'] = [int(center_x * ratio_x), int(center_y * ratio_y)]

        return annotation
    
    def get_next_frame_and_annotation(self):
        annotation = self.annotation[str(self.index)]
        self.index+=1
        frame = self.read()

        old_frame_shape = (frame.shape[1], frame.shape[0])
        frame = self.resize_frame(frame)
        new_frame_shape = (frame.shape[1],frame.shape[0])

        annotation = self.resize_annotation(annotation,old_frame_shape,new_frame_shape)

        return frame, annotation

def load_extension(extension_weights, intermediate_layer_size=32):
    extension_model = SurfNet(intermediate_layer_size)
    extension_model.load_state_dict(torch.load(extension_weights))
    for param in extension_model.parameters():
        param.requires_grad = False
    extension_model.to('cuda')
    extension_model.eval()
    return extension_model

def load_base(base_weights):
    base_model = create_base('dla_34', heads = {'hm':3,'wh':2}, head_conv=256)
    base_model = load_my_model(base_model, base_weights)
    for param in base_model.parameters():
        param.requires_grad = False 
    base_model.to('cuda')
    base_model.eval()
    return base_model

def base_extesnion(Module):

    def __init__(self, base_model, extension_model):
        self.base_model = base_model 
        self.extension_model = extension_model 

    def forward(self, x):

        x = self.base_model(x)[-1]['hm']
        x = torch.max(x, dim=1, keepdim=True)[0]
        return self.extension_model(x)

def compute_flow(frame0, frame1, downsampling_factor):
    h, w = frame0.shape[:-1]

    new_h = h // downsampling_factor
    new_w = w // downsampling_factor

    frame0 = cv2.resize(frame0, (new_w, new_h))
    frame1 = cv2.resize(frame1, (new_w, new_h))

    flow01 = flow_opencv_dense(frame0, frame1)
    
    return flow01

def detect(frame, threshold, base_model, extension_model):
    frame = transform_test_CenterNet()(frame).to('cuda').unsqueeze(0)
    base_result = torch.max(base_model(frame)[-1]['hm'], dim=1, keepdim=True)[0]
    extension_result = torch.sigmoid(extension_model(base_result))
    detections = nms(extension_result).gt(threshold).squeeze()
    return torch.nonzero(detections).cpu().numpy()

def main(args):

    base_model = load_base(args.base_weights)
    extension_model = load_extension(args.extension_weights, 32)

    detector = lambda frame : detect(frame, threshold=0.38, base_model=base_model, extension_model=extension_model)
    video = VideoOpenCV(args.input_video, fix_res=False, downsampling_factor=args.downsampling_factor)
    num_frames = video.num_frames

    
    with torch.no_grad():  
        frame0, annotation0 = video.get_next_frame_and_annotation()

        detections0 = detector(frame0)
        
        for frame_nb in range(1,num_frames):

            frame1, annotation1 = video.get_next_frame_and_annotation()
            detections1 = detector(frame1)
            flow01 = compute_flow(frame0, frame1, args.downsampling_factor)

            test = 0 

        return 0 


if __name__ == '__main__': 
    # import argparse 
    # parser = argparse.ArgumentParser(description='Tracking')
    # parser.add_argument('--input-video',type='str')
    # parser.add_argument('--threshold-unassigned-objects',type=float)
    # parser.add_argument('--confidence-threshold',type=float)
    # parser.add_argument('--base-weights',type=str)
    # parser.add_argument('--extension-weights',type=str)


    # args = parser.parse_args()

    class Args(object):
        def __init__(self, base_weights, extension_weights, input_video, threshold_unassigned_objects, confidence_threshold, downsampling_factor):
            self.base_weights = base_weights
            self.extension_weights = extension_weights 
            self.input_video = input_video
            self.threshold_unassigned_objects = threshold_unassigned_objects
            self.confidence_threshold = confidence_threshold
            self.downsampling_factor = downsampling_factor


    args = Args(base_weights='external_pretrained_models/centernet_pretrained.pth', 
                extension_weights='experiments/extension/surfnet32_alpha_2_beta_4_lr_1e-5_lr_reduced_epoch_15_multi_obj_no_obj/model_49.pth',
                input_video='data/generated_videos/adour_3_1_two_objects.MP4',
                threshold_unassigned_objects=0,
                confidence_threshold=0,
                downsampling_factor=4)


    main(args)
