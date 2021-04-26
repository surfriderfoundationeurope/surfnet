from train_base import get_dataset
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
class Args(object):
    def __init__(self, data_path, dataset, downsampling_factor, old_train):
        self.old_train = old_train
        self.downsampling_factor = downsampling_factor
        self.dataset = dataset 
        self.data_path = data_path

args = Args('data/surfrider_images','surfrider', 4, False)

dataset, num_classes = get_dataset(args.data_path, args.dataset, "train", args)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# fig.canvas.mpl_connect('key_press_event', press)
# def press(event):
#     global i
#     if event.key == '1':
#         print('Appending frame')
#         keeped_frames.append(frames[i % 100])
#     i += 1
#     imgplot.set_data(frames[i % 100])
#     fig.canvas.draw()

for image, target in dataloader: 
    image = np.transpose(image[0].numpy(), axes=[1, 2, 0]) * (0.229, 0.224, 0.225) +  (0.498, 0.470, 0.415)
    hm, h, w = target[0][0], target[0][1], target[0][2]
    fig, (ax0,ax1,ax2,ax3) = plt.subplots(1,4)

    ax0.imshow(image)
    ax1.imshow(hm,cmap='gray')
    ax2.imshow(h,cmap='gray')
    ax3.imshow(w,cmap='gray')
    plt.show()

    # test = 0