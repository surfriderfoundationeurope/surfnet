from common.datasets import SurfnetDatasetFlow
import cProfile, pstats, io
from pstats import SortKey
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import torch
def profile_dataset(dataset):

    pr = cProfile.Profile()

    pr.enable()

    next(iter(dataset))    

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

heatmaps_folder = "data/extracted_heatmaps/"
annotations_dir = 'data/generated_videos/'

video_dataset_train = SurfnetDatasetFlow(annotations_dir=annotations_dir, heatmaps_folder=heatmaps_folder, split='train')
video_dataset_test = SurfnetDatasetFlow(annotations_dir=annotations_dir, heatmaps_folder=heatmaps_folder, split='val')

video_loader_train = DataLoader(video_dataset_train, batch_size=1, shuffle=True)
video_loader_test = DataLoader(video_dataset_test, batch_size=1, shuffle=False)

for heatmap0, gt0 in video_loader_test:
    fig, (ax0, ax1, ax2) = plt.subplots(1,3)
    ax0.imshow(torch.sigmoid(heatmap0)[0][0],cmap='gray', vmin=0, vmax=1)
    ax1.imshow(gt0[0], cmap='gray', vmin=0, vmax=1)
    # ax2.imshow(gt1[0], cmap='gray',vmin=0,vmax=1)
    plt.show()
    plt.close()


