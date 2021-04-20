from base.deeplab.models import get_model
from common.datasets import SingleVideoDataset
import pickle 
import os 
from base.centernet.models import create_model as create_model_centernet
from base.centernet.models import load_model as load_model_centernet
from common.utils import blob_for_bbox, pre_process_centernet, transform_test_CenterNet, transforms_test_deeplab
import numpy as np 
import matplotlib.pyplot as plt
from common.utils import load_my_model
import torch
import cv2 
import json
from synthetic_videos.flow_tools import flow_opencv_dense
from PIL import Image
from tqdm import tqdm

def plot_single_image_heatmaps_and_gt(image, heatmaps, gt, normalize):

    _ , (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1,5,figsize=(30,30))
    ax0.imshow(image)
    if not normalize: 
        kwargs = {'vmin':0, 'vmax':1, 'cmap':'gray'}
    else: 
        kwargs = {'cmap':'gray'}

    ax1.imshow(heatmaps[0].sigmoid_(), **kwargs)
    ax2.imshow(heatmaps[1].sigmoid_(), **kwargs)
    ax3.imshow(heatmaps[2].sigmoid_(), **kwargs)
    ax4.imshow(gt, **kwargs)

    plt.show()

class VideoOpenCV(object):

    def __init__(self, video_name, fix_res=False, downsampling_factor=4):
        self.cap = cv2.VideoCapture(video_name)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fix_res = fix_res
        self.downsampling_factor = downsampling_factor
        # self.annotation = json.load(open(video_name.replace('.MP4','.json') ,'r'))

    def read(self):
        ret, frame = self.cap.read()
        old_frame_shape = (frame.shape[1], frame.shape[0])

        if not ret: 
            print('Unreadable frame!')
        frame = self.resize_frame(frame)
        new_frame_shape = (frame.shape[1], frame.shape[0])
        return frame, old_frame_shape, new_frame_shape

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

    # def resize_annotation(self, annotation, old_frame_shape, new_frame_shape):

    #     old_shape_x, old_shape_y = old_frame_shape
    #     new_shape_x, new_shape_y = new_frame_shape

    #     ratio_x = new_shape_x / old_shape_x
    #     ratio_y = new_shape_y / old_shape_y

    #     for object_nb in annotation.keys():

    #         [top_left_x, top_left_y, width, height] = annotation[object_nb]['bbox']
    #         [center_x, center_y] = annotation[object_nb]['center']

    #         annotation[object_nb]['bbox'] = [int(top_left_x * ratio_x), int(top_left_y * ratio_y), int(width * ratio_x), int(height * ratio_y)]
    #         annotation[object_nb]['center'] = [int(center_x * ratio_x), int(center_y * ratio_y)]

    #     return annotation
    
    # def get_next_frame_and_annotation(self):
    #     annotation = self.annotation[str(self.index)]
    #     self.index+=1
    #     frame = self.read()

    #     old_frame_shape = (frame.shape[1], frame.shape[0])
    #     frame = self.resize_frame(frame)
    #     new_frame_shape = (frame.shape[1],frame.shape[0])

    #     annotation = self.resize_annotation(annotation,old_frame_shape,new_frame_shape)

    #     return frame, annotation

# def build_gt(annotation, shape, downsampling_factor):
#     shape = (shape[0] // downsampling_factor, shape[1] // downsampling_factor)

#     Phi = np.zeros(shape=shape)

#     for object_nb in annotation:
#         Phi = np.maximum(Phi, blob_for_bbox(annotation[str(object_nb)]['bbox'], Phi, downsampling_factor)[0])

#     return Phi

def save_heatmap(folder_for_video, frame_nb, heatmap):
    with open(folder_for_video + '{:03d}.pickle'.format(frame_nb), 'wb') as f:
        pickle.dump(heatmap, f)
        
def save_flow(folder_for_video, frame_nb, flow):
    output_name = folder_for_video + '{:03d}_{:03d}'.format(frame_nb-1, frame_nb)
    np.save(output_name, flow)

def _get_heatmap(frame, model, transform):
    frame = transform(frame).to('cuda').unsqueeze(0)
    return model(frame)[-1]['hm'].cpu().squeeze()

def compute_flow(frame0, frame1, downsampling_factor):
    h, w = frame0.shape[:-1]

    new_h = h // downsampling_factor
    new_w = w // downsampling_factor

    frame0 = cv2.resize(frame0, (new_w, new_h))
    frame1 = cv2.resize(frame1, (new_w, new_h))

    flow01 = flow_opencv_dense(frame0, frame1)
    
    return flow01

def verbose(frame0, frame1,  heatmap0, heatmap1, flow01):

    fig, ((ax0,ax1), (ax4, ax5), (ax6, ax7)) = plt.subplots(3,2)

    frame0 = Image.fromarray(cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB))
    frame1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))

    ax0.imshow(frame0)
    ax0.set_title('Frame 0')

    ax1.imshow(frame1)
    ax1.set_title('Frame 1')

    ax4.imshow(torch.sigmoid(torch.max(heatmap0,dim=0)[0]), cmap='gray', vmin=0, vmax=1)
    ax4.set_title('Heatmap 0')

    ax5.imshow(torch.sigmoid(torch.max(heatmap1,dim=0)[0]), cmap='gray', vmin=0, vmax=1)    
    ax5.set_title('Heatmap 1')


    mag, ang = cv2.cartToPolar(flow01[...,0], flow01[...,1])
    hsv = np.zeros(shape=(*flow01.shape[:-1],3),dtype=np.uint8)
    hsv[...,1] = 0
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    flow_rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    ax6.imshow(flow_rgb)
    ax6.set_title('Flow between 0 and 1')

    ax7.set_axis_off()

    plt.show()
    plt.close()

def resize_annotations(annotations, old_shape, new_shape, downsampling_factor):

    old_shape_x, old_shape_y = old_shape
    new_shape_x, new_shape_y = new_shape

    ratio_x = new_shape_x / old_shape_x
    ratio_y = new_shape_y / old_shape_y

    for annotation in annotations:

        [top_left_x, top_left_y, width, height] = annotation['bbox']
        new_bbox = [top_left_x * ratio_x, top_left_y * ratio_y, width * ratio_x, height * ratio_y]
        annotation['bbox'] = [new_bbox_coord // downsampling_factor for new_bbox_coord in new_bbox]

    return annotations

def extract_heatmaps_for_video_frames(model, transform, args):

    video_folder = args.input_dir
    video_names = [video_name for video_name in sorted(os.listdir(video_folder))[:2] if '.MP4' in video_name]

    get_heatmap = lambda frame: _get_heatmap(frame, model, transform)

    with torch.no_grad():
        for video_name in tqdm(video_names): 
            folder_for_video = args.output_dir + video_name.strip('.MP4') +'/'
            os.mkdir(folder_for_video)
            # print('Processing video {}'.format(video_nb))
            video = VideoOpenCV(video_folder + video_name, fix_res=False, downsampling_factor=args.downsampling_factor)
            num_frames = video.num_frames
            frame_nb = 1

            frame0, old_shape, new_shape = video.read()
            heatmap0 = get_heatmap(frame0)
            save_heatmap(folder_for_video, frame_nb, heatmap0)

            for frame_nb in range(2,num_frames+1):

                frame1, _ , _  = video.read()
                heatmap1 = get_heatmap(frame1)
                save_heatmap(folder_for_video, frame_nb, heatmap1)

                flow01 = compute_flow(frame0, frame1, args.downsampling_factor)
                save_flow(folder_for_video, frame_nb, flow01)
                # verbose(frame0, frame1, heatmap0, heatmap1, flow01)

                frame0 = frame1.copy()
    with open(args.input_dir+'annotations.json','r') as f:
        COCO_formatted_annotations_old = json.load(f)
    
    COCO_formatted_annotations_new = COCO_formatted_annotations_old.copy()
    COCO_formatted_annotations_new['annotations'] = resize_annotations(COCO_formatted_annotations_old['annotations'],old_shape, new_shape, args.downsampling_factor)
    with open(args.input_dir+'annotations_resized.json','w') as f:
        json.dump(COCO_formatted_annotations_new, f)

def extract(args):

    trained_model_weights_filename = args.weights

    if 'deeplab' in args.model:
        transform = transforms_test_deeplab
        model = get_model(args.model, 4, freeze_backbone=True, downsampling_factor=args.downsampling_factor, focal=False)
        model = load_my_model(model, trained_model_weights_filename)

    else:
        if args.my_repo: 
            transform = transform_test_CenterNet(fix_res=False)
            model = create_model_centernet(arch=args.model, heads={'hm':3,'wh':2}, head_conv=256)
            model = load_my_model(model, trained_model_weights_filename)

        else:
            transform = pre_process_centernet
            model = create_model_centernet(arch=args.model, heads={'hm':3,'wh':2,'reg':2}, head_conv=256)
            model = load_model_centernet(model, trained_model_weights_filename)

    for param in model.parameters():
        param.requires_grad = False
    model.to('cuda')
    print('Model loaded to GPU.')
    model.eval()

    if args.from_videos:
        extract_heatmaps_for_video_frames(model, transform, args)
    
    # else:
    #     extract_heatmaps_for_images_in_folder(model, transform, args)


if __name__ == '__main__':


    import argparse
    
    parser = argparse.ArgumentParser(description='Extracting heatmaps produced by base network')

    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--downsampling-factor', type=int)
    parser.add_argument('--model', type=str)
    parser.add_argument('--my-repo', action='store_true')
    parser.add_argument('--from-videos', action='store_true')
    parser.add_argument('--extract-flow', action='store_true')

    args = parser.parse_args()
    # print(args)
    # return 0
    extract(args)
    
    




        



