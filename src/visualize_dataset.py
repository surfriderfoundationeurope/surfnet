import numpy as np
import matplotlib.pyplot as plt
from train_detector import get_dataset

plt.ion()
fig, (ax0, ax1, ax2, ax3) = plt.subplots(1,4)

class Args(object):
    def __init__(self, data_path, dataset, downsampling_factor, old_train):
        self.old_train = old_train
        self.downsampling_factor = downsampling_factor
        self.dataset = dataset
        self.data_path = data_path


args = Args('./data/images','surfrider', downsampling_factor=4, old_train=False)


dataset, num_classes = get_dataset(args.data_path, args.dataset, "train", args)

for image, target in dataset: 
    image = np.transpose(image.numpy(), axes=[1, 2, 0]) * (0.229, 0.224, 0.225) +  (0.498, 0.470, 0.415)
    ax0.imshow(image)
    ax1.imshow(target[0],cmap='gray',vmin=0,vmax=1)
    ax2.imshow(target[1],cmap='gray')
    ax3.imshow(target[2],cmap='gray')
    plt.show()
    while not plt.waitforbuttonpress(): continue 
    plt.cla()
