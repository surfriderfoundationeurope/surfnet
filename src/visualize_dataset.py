from common.datasets.datasets import SurfnetDataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch

plt.ion()
fig, (ax0, ax1) = plt.subplots(1,2)

class Args(object):
    def __init__(self, data_path, dataset, downsampling_factor, old_train):
        self.old_train = old_train
        self.downsampling_factor = downsampling_factor
        self.dataset = dataset
        self.data_path = data_path


args = Args('./data/surfrider_images/','surfrider', downsampling_factor=1, old_train=False)

from train_base import get_dataset

dataset, num_classes = get_dataset(args.data_path, args.dataset, "train", args)

for image, target in dataset: 
    image = np.transpose(image.numpy(), axes=[1, 2, 0]) * (0.229, 0.224, 0.225) +  (0.498, 0.470, 0.415)
    target = target[0]
    ax0.imshow(image)
    ax1.imshow(target,cmap='gray',vmin=0,vmax=1)
    plt.show()
    while not plt.waitforbuttonpress(): continue 
    plt.cla()

# args = Args('data/surfrider_images','surfrider', 4, False)


# dataset = SurfnetDataset('/home/infres/chagneux/repos/surfnet/data/synthetic_videos_dataset/annotations',
#                          '/home/infres/chagneux/repos/surfnet/data/extracted_heatmaps/dla_34_downsample_4_alpha_2_beta_4_lr_6.25e-5_single_class', split='train')

# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# for heatmap0, target0, target1, flows01 in dataloader:
#     # image = np.transpose(image[0].numpy(), axes=[1, 2, 0]) * (0.229, 0.224, 0.225) +  (0.498, 0.470, 0.415)
#     # hm, h, w = target[0][0], target[0][1], target[0][2]
#     fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

#     ax0.imshow(torch.sigmoid(heatmap0[0][0]), cmap='gray')
#     ax1.imshow(target0[0][0], cmap='gray')
#     ax2.imshow(target1[0][0], cmap='gray')
#     plt.show()

#     test = 0

# import pickle
# old_shape = (1080,1920)
# new_shape = (272,488)
# with open('data/extracted_heatmaps/dla_34_downsample_4_alpha_2_beta_4_lr_6.25e-5_single_class/shapes.pickle','wb') as f:
#     data =  (old_shape, new_shape)
#     pickle.dump(data,f)
