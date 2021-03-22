import torch.utils.data
import os 
import numpy as np
import json 
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm as tqdm
import torch 
import pickle
from torch.nn.functional import sigmoid

class SingleVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_name, transforms = None):
        self.video = torchvision.io.read_video(video_name)[0].permute(0,3,1,2)
        self.annotation = json.load(open(video_name.replace('.MP4','.json') ,'r'))
        self.transforms = transforms 
        self.PILConverter = torchvision.transforms.ToPILImage()

    def __getitem__(self, index):

        frame = self.PILConverter(self.video[index])
        old_frame_shape = frame.size
        
        annotation = self.annotation[str(index)]

        if self.transforms is not None: 
            frame =  self.transforms(frame)
            new_frame_shape = frame.shape

            annotation = self._resize_annotation(annotation, old_frame_shape, new_frame_shape)

        return frame, annotation

    def _resize_annotation(self, annotation, old_frame_shape, new_frame_shape):

        old_shape_x, old_shape_y = old_frame_shape
        new_shape_x, new_shape_y = new_frame_shape[2], new_frame_shape[1]

        ratio_x = new_shape_x / old_shape_x
        ratio_y = new_shape_y / old_shape_y

        for object_nb in annotation.keys():

            [top_left_x, top_left_y, width, height] = annotation[object_nb]['bbox']
            [center_x, center_y] = annotation[object_nb]['center']

            annotation[object_nb]['bbox'] = [int(top_left_x * ratio_x), int(top_left_y * ratio_y), int(width * ratio_x), int(height * ratio_y)]
            annotation[object_nb]['center'] = [int(center_x * ratio_x), int(center_y * ratio_y)]

        return annotation

    def __len__(self):
        return len(self.video)      
 
class SurfnetDataset(torch.utils.data.Dataset):
    def __init__(self, heatmaps_folder, split):
        self.heatmaps_folder = heatmaps_folder
        self.split = split 
        self.subset = False
        self.heatmap_filenames = sorted(os.listdir(heatmaps_folder))
        self._init_ids()



    def __getitem__(self, index):

        if self.split == 'train':
            video_id, pair_id_in_video = self.id_to_location[index]
            video_name_0 = 'video_{:03d}_frame_{:03d}.pickle'.format(video_id, pair_id_in_video)
            video_name_1 = 'video_{:03d}_frame_{:03d}.pickle'.format(video_id, pair_id_in_video+1)
            with open(self.heatmaps_folder + video_name_0,'rb') as f: 
                Z_0, Phi_0_tilde, center_0 = pickle.load(f)
            with open(self.heatmaps_folder + video_name_1,'rb') as f: 
                Phi_1_tilde, center_1 = pickle.load(f)[1:]

            d_01 = np.array(center_0) - np.array(center_1)

            return Z_0, Phi_0_tilde, Phi_1_tilde, d_01
        else: 
            video_id, id_in_video = self.id_to_location[index]
            video_name = 'video_{:03d}_frame_{:03d}.pickle'.format(video_id, id_in_video)

            with open(self.heatmaps_folder + video_name,'rb') as f: 
                 Z, Phi_tilde, _ = pickle.load(f)

            return Z, Phi_tilde

    def __len__(self):
        return len(self.id_to_location)

    def _init_ids(self):

        self.ids_dict = dict()

        for filename in self.heatmap_filenames:
            splitted_name = filename.split('_')
            video_id, frame_id_in_video = splitted_name[1], splitted_name[3].strip('.pickle')
            self.ids_dict.setdefault(video_id, []).append(frame_id_in_video) 
        
        self.id_to_location = []
        num_videos = len(self.ids_dict.keys())
        num_videos_train = int(0.9*num_videos)

        if self.split == 'train':
            for video_id in list(self.ids_dict.keys())[:num_videos_train]:
                for frame_id_in_video in self.ids_dict[video_id][:-1]:
                    self.id_to_location.append((int(video_id), int(frame_id_in_video)))
        
        else:
            for video_id in list(self.ids_dict.keys())[num_videos_train:]:
                for frame_id_in_video in self.ids_dict[video_id]:
                    self.id_to_location.append((int(video_id), int(frame_id_in_video)))

        if self.subset: 
            self.id_to_location = self.id_to_location[:int(0.1*len(self.id_to_location))]

def profile_dataset(dataset):
    import cProfile, pstats, io
    from pstats import SortKey
    pr = cProfile.Profile()

    pr.enable()

    next(iter(dataset))    

    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

def plot_loader(video_loader):

    if video_loader.dataset.split == 'train':

        Z_0s, Phi_0_tildes, Phi_1_tildes, d_01s = next(iter(video_loader))
        
        Z_0s_translated = spatial_transformer(Z_0s, d_01s)

        for Z_0, Z_0_translated, Phi_0_tilde, Phi_1_tilde, d_01 in zip(Z_0s[:,0,:,:], Z_0s_translated[:,0,:,:], Phi_0_tildes[:,0,:,:], Phi_1_tildes[:,0,:,:], d_01s):


        # plt.hist(Phi_0_tilde)
        # plt.show()

            fig, ((ax0, ax1), (ax2, ax3), (ax4, ax5)) = plt.subplots(3,2, figsize=(10,10))
            ax0.imshow(Z_0, cmap='gray')
            ax0.set_title('$Z_0$')

            ax1.imshow(Z_0_translated, cmap='gray')
            ax1.set_title('$T(Z_0,d_{01})$')

            ax2.imshow(Phi_0_tilde, cmap='gray')
            ax2.set_title('$\widetilde{\Phi}_0$')

            ax3.imshow(Phi_1_tilde, cmap='gray')
            ax3.set_title('$\widetilde{\Phi}_1$')
            
            ax4.imshow(sigmoid(Phi_0_tilde), cmap='gray')
            ax4.set_title('$\Phi_0$')

            ax5.imshow(sigmoid(Phi_1_tilde), cmap='gray')
            ax5.set_title('$\Phi_1$')
            
             
            plt.suptitle('$d_{01} = $'+str(d_01.numpy()))

            plt.show()

    else: 

        Zs, Phi_tildes = next(iter(video_loader))
        for Z, Phi_tilde in zip(Zs[:,0,:,:],Phi_tildes):
            fig, (ax0, ax1) = plt.subplots(1,2)
            ax0.imshow(Z, cmap='gray')
            ax1.imshow(Phi_tilde, cmap='gray')
            plt.show()
            
if __name__ == "__main__":

    heatmaps_folder = "/media/mathis/f88b9c68-1ae1-4ecc-a58e-529ad6808fd3/heatmaps_and_annotations/"

    video_dataset = SurfnetDataset(heatmaps_folder, split='train')

    profile_dataset(video_dataset)

    # video_loader = torch.utils.data.DataLoader(video_dataset,batch_size=8,shuffle=True)

    # plot_loader(video_loader)


