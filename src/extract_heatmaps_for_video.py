from tools.misc import load_model 
import os 
from tools.video_readers import IterableFrameReader
from detection.detect import transform_for_test, nms
import cv2
import torch 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
plt.ion()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model_weights = 'experiments/detection/retrain_3500_images_no_DCNv2/model_70.pth' 
data_dir = 'data/validation_videos'
skipped_frames = 2300

print('Loading model')
model = load_model(model_weights)
print('Model loaded')
video_filenames = [video_filename for video_filename in os.listdir(data_dir) if video_filename.endswith('.mp4')]

fig, (ax0, ax1, ax2) = plt.subplots(1,3)
for video_filename in video_filenames:
    reader = IterableFrameReader(os.path.join(data_dir,video_filename), skip_frames=3, output_shape=(960,544))
    reader.set_head(skipped_frames)
    for frame_nb, frame in enumerate(reader): 
        heatmap =  torch.sigmoid(model(transform_for_test()(frame).to('cuda').unsqueeze(0))[-1]['hm']).squeeze().cpu()
        heatmap2 = nms(heatmap.unsqueeze(0).unsqueeze(0)).squeeze().numpy()
        heatmap = heatmap.numpy()
        heatmap2[heatmap2 < 0.4] = 0
        print(skipped_frames+frame_nb)
        ax0.imshow(cv2.cvtColor(cv2.resize(frame,heatmap.shape[::-1]),cv2.COLOR_BGR2RGB))
        ax1.imshow(heatmap,cmap='gray',vmin=0,vmax=1)
        ax2.imshow(heatmap2,cmap='gray',vmin=0,vmax=1)
        plt.tight_layout()
        plt.autoscale(True)
        while not plt.waitforbuttonpress(): continue 
        plt.cla()
        
        # cv2.imshow('heatmap',cv2.resize(heatmap,(1280,720)))
        # cv2.imshow('thresholded_heatmap',cv2.resize(heatmap2,(1280,720)))




