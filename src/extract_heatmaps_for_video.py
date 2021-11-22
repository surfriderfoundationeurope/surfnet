from tools.misc import load_model 
import os 
from tools.video_readers import IterableFrameReader
from detection.detect import transform_for_test, nms
from tools.optical_flow import compute_flow
from warp_flow import warp_flow
import cv2
import torch 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import numpy as np 

plt.ion()
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
model_weights = 'experiments/detection/retrain_3500_images_no_DCNv2/model_114.pth' 
video_filename = '/home/infres/chagneux/repos/surfnet/data/validation_videos/T1/full/T1_1080_px_converted.mp4'
skipped_frames = 0
nb_frames = 4


reader = IterableFrameReader(video_filename, skip_frames=5, output_shape=(960,544))
reader.set_head(skipped_frames)

frames = []
for _ in range(nb_frames):
    frames.append(next(reader))
flows = []
for frame0, frame1 in tqdm(zip(frames[:-1],frames[1:])):
    flows.append(compute_flow(frame0,frame1,downsampling_factor=4))

# frames_to_warp = torch.from_numpy(np.array(frames[:-1])).permute(0,3,1,2) / 255.
# flows_for_warping = - torch.from_numpy(np.array(flows))
# test = 0 
# warped_frames = warp_flow(frames_to_warp, flows_for_warping,device = 'cpu')
# warped_frames = (warped_frames * 255.).permute(0,2,3,1).numpy() 
# warped_frames = warped_frames.astype(np.uint8)

device = torch.device('cpu')
model = load_model(model_weights, device)

batched_frames_for_model = torch.stack([transform_for_test()(frame) for frame in frames])
heatmaps = torch.sigmoid(model(batched_frames_for_model)[-1]['hm'])

flows_for_warping = - torch.from_numpy(np.array(flows))
warped_heatmaps = warp_flow(heatmaps[:-1], flows_for_warping, device)
heatmaps = heatmaps.cpu()
warped_heatmaps = warped_heatmaps.cpu()

fig, (ax0, ax1, ax2) = plt.subplots(1,3)

for frame1, heatmap0, heatmap1, warped_heatmap0 in zip(frames[1:], heatmaps[:-1], heatmaps[1:], warped_heatmaps):


    new_stuff = heatmap1 - warped_heatmap0
    new_stuff[new_stuff < 0] = 0

    gone_stuff = warped_heatmap0 - heatmap1
    gone_stuff[gone_stuff < 0] = 0

    ax0.imshow(cv2.cvtColor(frame1,cv2.COLOR_BGR2RGB))
    ax1.imshow(heatmap0.squeeze(), cmap='gray', vmin=0, vmax=1)
    ax2.imshow(heatmap1.squeeze(), cmap='gray', vmin=0, vmax=1)
    heatmap_1_sum = heatmap1.sum()
    heatmap_0_sum = heatmap0.sum()
    plt.suptitle(f'{heatmap_1_sum/heatmap_0_sum - 1:4f}')

    ax0.set_title('Frame k+1')
    ax1.set_title(f'Heatmap k, sum = {heatmap_0_sum:.4f}')
    ax2.set_title(f'Heatmap k+1, sum = {heatmap_1_sum:.4f}')

    plt.tight_layout()
    plt.autoscale(True)
    while not plt.waitforbuttonpress(): continue 
    plt.cla()
    



