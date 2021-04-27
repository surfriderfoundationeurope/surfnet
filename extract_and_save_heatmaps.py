from base.deeplab.models import get_model
from train_extension import spatial_transformer
import pickle 
import os 
from base.centernet.models import create_model as create_model_centernet
from base.centernet.models import load_model as load_model_centernet
from common.utils import pre_process_centernet, transform_test_CenterNet, transforms_test_deeplab
import numpy as np 
import matplotlib.pyplot as plt
from common.utils import load_my_model
import torch
import cv2 
import json
from synthetic_videos.flow_tools import flow_opencv_dense
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import ToTensor, ToPILImage


def save_heatmap(heatmaps_dir, frame_nb, heatmap):
    filename = os.path.join(heatmaps_dir,'{:03d}.pickle'.format(frame_nb+1))
    with open(filename, 'wb') as f:
        pickle.dump(heatmap, f)
        
def save_flow(flow_dir, frame_nb, flow):
    output_name = os.path.join(flow_dir,'{:03d}_{:03d}'.format(frame_nb, frame_nb+1))
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
    h,w = flow01.shape[:-1]
# 
    frame0 = Image.fromarray(cv2.cvtColor(cv2.resize(frame0,(w,h)), cv2.COLOR_BGR2RGB))
    frame1 = Image.fromarray(cv2.cvtColor(cv2.resize(frame1,(w,h)), cv2.COLOR_BGR2RGB))


    frame0_to_warp = ToTensor()(frame0).unsqueeze(0).to('cuda')
    flow01_to_warp = torch.tensor(flow01).unsqueeze(0).to('cuda')
    frame0_warped = spatial_transformer(frame0_to_warp,-flow01_to_warp,device='cuda')
    ax0.imshow(frame0)
    ax0.set_title('Frame 0')

    ax1.imshow(frame1)
    ax1.set_title('Frame 1')

    ax4.imshow(torch.sigmoid(heatmap0), cmap='gray', vmin=0, vmax=1)
    ax4.set_title('Heatmap 0')

    ax5.imshow(torch.sigmoid(heatmap1), cmap='gray', vmin=0, vmax=1)    
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

    ax7.imshow(ToPILImage()(frame0_warped.squeeze().cpu()))
    ax7.set_title('Frame 0 warped with flow')

    plt.show()
    plt.close()


def save_frame(frames_dir, frame_nb, frame):
    cv2.imwrite(frames_dir + '{:03d}.png'.format(frame_nb), frame)

def read_and_resize(filename):
    frame = cv2.imread(filename)
    h, w = frame.shape[:-1]
    new_h = (h | 31) + 1
    new_w = (w | 31) + 1
    frame = cv2.resize(frame, (new_w, new_h))
    new_shape = (new_h, new_w)
    old_shape = (h,w)
    return frame, old_shape, new_shape

def extract_heatmaps_for_video(filenames, heatmaps_dir_for_video, flows_dir_for_video, model, transform):

    get_heatmap = lambda frame: _get_heatmap(frame, model, transform)

    with torch.no_grad():
        # print('Processing video {}'.format(video_nb))
        num_frames = len(filenames)

        frame0, old_shape, new_shape = read_and_resize(filenames[0])


        heatmap0 = get_heatmap(frame0)
        save_heatmap(heatmaps_dir_for_video, 0, heatmap0)

        for frame_nb in range(1, num_frames):
            
            frame1, _ ,_ = read_and_resize(filenames[frame_nb])
    
            flow01 = compute_flow(frame0, frame1, args.downsampling_factor)
            save_flow(flows_dir_for_video, frame_nb, flow01)

            heatmap1 = get_heatmap(frame1)
            save_heatmap(heatmaps_dir_for_video, frame_nb, heatmap1)

            verbose(frame0, frame1, heatmap0, heatmap1, flow01)
            frame0 = frame1.copy()
            
    return old_shape, new_shape


def extract(args):

    trained_model_weights_filename = args.weights

    transform = transform_test_CenterNet()
    model = create_model_centernet(arch='dla_34', heads={'hm':1,'wh':2}, head_conv=256)
    model = load_my_model(model, trained_model_weights_filename)

    for param in model.parameters():
        param.requires_grad = False
    model.to('cuda')
    print('Model loaded to GPU.')
    model.eval()

    output_dir = os.path.join(args.output_dir,trained_model_weights_filename.split('/')[-2])
    os.mkdir(output_dir)
    flows_dir = os.path.join(output_dir,'flows')
    os.mkdir(flows_dir)
    heatmaps_dir = os.path.join(output_dir,'heatmaps')
    os.mkdir(heatmaps_dir)

    annotation_filename = os.path.join(args.dataset_dir,'annotations','annotations.json')
    with open(annotation_filename,'r') as f:
        annotations = json.load(f)
    
    for video in tqdm(annotations['videos']):
        flows_dir_for_video = os.path.join(flows_dir, video['file_name'])
        os.mkdir(flows_dir_for_video)

        heatmaps_dir_for_video = os.path.join(heatmaps_dir, video['file_name'])
        os.mkdir(heatmaps_dir_for_video)

        images = [image for image in annotations['images'] if image['video_id'] == video['id']]
        images = sorted(images,key=lambda image:image['frame_id'])
        filenames = [os.path.join(args.dataset_dir,'data',image['file_name']) for image in images]

        old_shape, new_shape = extract_heatmaps_for_video(filenames, heatmaps_dir_for_video, flows_dir_for_video, model, transform)
    
    with open(os.path.join(output_dir,'shapes.pickle'),'wb') as f:
        new_shape[0] = new_shape[0] // args.downsampling_factor
        new_shape[1] = new_shape[1] // args.downsampling_factor
        data = (old_shape, new_shape)
        pickle.dump(data,f)

if __name__ == '__main__':


    import argparse
    
    parser = argparse.ArgumentParser(description='Extracting heatmaps produced by base network')

    parser.add_argument('--dataset_dir')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--downsampling_factor', type=int)

    args = parser.parse_args()
    extract(args)
    
    




        



