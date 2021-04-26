import torch.utils.data
import os 
import numpy as np
import json 
import matplotlib.pyplot as plt
import torchvision
from tqdm import tqdm as tqdm
import torch 
import pickle
# import math 
import cv2
from common.utils import blob_for_bbox

class VideoOpenCV(object):
    def __init__(self, video_name):
        self.cap = cv2.VideoCapture(video_name)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self):
        ret, frame = self.cap.read()
        if not ret: 
            print('Unreadable frame!')
        return frame

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
        self.heatmap_filenames = [filename for filename in sorted(os.listdir(heatmaps_folder)) if filename.endswith('.pickle')]
        self._init_ids()



    def __getitem__(self, index):

        if self.split == 'train':
            video_id, pair_id_in_video = self.id_to_location[index]
            video_name_0 = 'video_{:03d}_frame_{:03d}.pickle'.format(video_id, pair_id_in_video)
            video_name_1 = 'video_{:03d}_frame_{:03d}.pickle'.format(video_id, pair_id_in_video+1)
            flow_name = 'flow_video_{:03d}_frame_{:03d}_{:03d}'.format(video_id,pair_id_in_video, pair_id_in_video+1)
            with open(self.heatmaps_folder + video_name_0,'rb') as f: 
                Z_0, Phi0, center_0 = pickle.load(f)
            with open(self.heatmaps_folder + video_name_1,'rb') as f: 
                Phi1, center_1 = pickle.load(f)[1:]
            flow01 = np.load(flow_name)

            d_01 = np.array(center_1) - np.array(center_0)

            Z_0 = torch.max(Z_0, axis=0, keepdim=True)[0]
            return  Z_0, Phi0, Phi1, d_01
        else: 
            video_id, id_in_video = self.id_to_location[index]
            video_name = 'video_{:03d}_frame_{:03d}.pickle'.format(video_id, id_in_video)

            with open(self.heatmaps_folder + video_name,'rb') as f: 
                 Z, Phi, _ = pickle.load(f)

            Z = torch.max(Z, axis=0, keepdim=True)[0]

            return Z, Phi

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

class SurfnetDatasetFlow(torch.utils.data.Dataset):

    def __init__(self, annotations_dir, heatmaps_folder, split='val'):
        self.split = split
        if self.split == 'train':
            annotation_filename = annotations_dir + 'annotations_train.json'
        else: 
            annotation_filename = annotations_dir + 'annotations_val.json'

        with open(annotation_filename, 'r') as f: 
            self.annotations = json.load(f)

        self.heatmaps_folder = heatmaps_folder

        self.subset = False

        self.init_ids()


    def __getitem__(self, index):

        location_id = self.ids[index]

        if self.split == 'train':
            heatmap0, gt0, gt1, flow01 = self.get_training_item(location_id)

            Z_0 = torch.max(heatmap0, axis=0, keepdim=True)[0]
            
            return Z_0, gt0, gt1, torch.from_numpy(flow01)

        else: 
            heatmap, gt = self.get_testing_item(location_id)

            Z = torch.max(heatmap, axis=0, keepdim=True)[0]

            return Z, gt

    def __len__(self):
        return len(self.ids)

    def init_ids(self):
        organised_annotations = dict()
        self.id_to_location = []
        for video in self.annotations['videos']:
            images_from_video = [image for image in self.annotations['images'] if image['video_id']==video['id']]
            organised_annotations[video['id']] = []
            for image in sorted(images_from_video,key=lambda x:x['frame_id']):
                annotations_for_image = [annotation for annotation in self.annotations['annotations'] if annotation['image_id'] == image['id']]
                organised_annotations[video['id']].append((image,annotations_for_image))
        self.annotations = organised_annotations
        
        self.ids = []
        if self.split == 'train':
            for video_id in self.annotations.keys():
                for frame in range(len(self.annotations[video_id])-1):
                    self.ids.append((video_id,frame,frame+1))

        else:
            for video_id in self.annotations.keys():
                for frame in range(len(self.annotations[video_id])):
                    self.ids.append((video_id,frame))
    def get_training_item(self, location_id):
        
        image0, annotations0 = self.annotations[location_id[0]][location_id[1]]
        image1, annotations1 = self.annotations[location_id[0]][location_id[2]]
        frame_id_0 = image0['frame_id']
        frame_id_1 = image1['frame_id']
        folder_name = self.heatmaps_folder + image0['file_name'].split('.')[0]
        with open(folder_name+'/{:03d}.pickle'.format(frame_id_0),'rb') as f:
            heatmap0 = pickle.load(f)
        shape = heatmap0.shape

        flow01 = np.load(folder_name+'/{:03d}_{:03d}.pickle'.format(frame_id_0,frame_id_1))
        bboxes0 = [annotation['bbox'] for annotation in annotations0]
        bboxes1 = [annotation['bbox'] for annotation in annotations1]
        gt0 = self.build_gt(bboxes0, shape)
        gt1 = self.build_gt(bboxes1,shape)

        return flow01, heatmap0, gt0, gt1
    
    def get_testing_item(self, location_id):

        image, annotations = self.annotations[location_id[0]][location_id[1]]
        frame_id = image['frame_id']
        folder_name = self.heatmaps_folder + image['file_name'].split('.')[0]
        with open(folder_name+'/{:03d}.pickle'.format(frame_id),'rb') as f:
            heatmap = pickle.load(f)

        shape = heatmap.shape
        bboxes = [annotation['bbox'] for annotation in annotations]
        gt = self.build_gt(bboxes, shape)

        return heatmap, gt
    

            
    def build_gt(self, bboxes, shape):

        gt = np.zeros(shape=shape)
        for bbox in bboxes:
            gt = np.maximum(gt, blob_for_bbox(bbox, gt)[0]) 

        return gt


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
            
            ax4.imshow(torch.sigmoid(Phi_0_tilde), cmap='gray')
            ax4.set_title('$\Phi_0$')

            ax5.imshow(torch.sigmoid(Phi_1_tilde), cmap='gray')
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
            


    # profile_dataset(video_dataset)

    # video_loader = torch.utils.data.DataLoader(video_dataset,batch_size=8,shuffle=True)

    # plot_loader(video_loader)


# class SingleVideoDataset(torch.utils.data.Dataset):
#     def __init__(self, video_name, transforms = None):
#         self.video = VideoOpenCV(video_name)
#         self.annotation = json.load(open(video_name.replace('.MP4','.json') ,'r'))
#         self.transforms = transforms 
        
#     def __getitem__(self, index):

#         frame = self.video.read()
#         old_frame_shape = frame.shape[:-1]
        
#         annotation = self.annotation[str(index)]

#         if self.transforms is not None: 
#             frame =  self.transforms(frame)
#             new_frame_shape = frame.shape[1:]

#             annotation = self._resize_annotation(annotation, old_frame_shape, new_frame_shape)

#         return frame, annotation

#     def _resize_annotation(self, annotation, old_frame_shape, new_frame_shape):

#         old_shape_y, old_shape_x = old_frame_shape
#         new_shape_y, new_shape_x = new_frame_shape
#         ratio = new_shape_x / old_shape_x
#         padding_y = (new_shape_y - ratio*old_shape_y) // 2

#         # padding_top = abs(new_shape_y - old_shape_y) // 2
#         # padding_left =  abs(new_shape_x - old_shape_x) // 2

#         for object_nb in annotation.keys():

#             [top_left_x, top_left_y, width, height] = annotation[object_nb]['bbox']
#             [center_x, center_y] = annotation[object_nb]['center']

#             annotation[object_nb]['bbox'] = [int(ratio*top_left_x), int(ratio*top_left_y + padding_y), int(ratio*width), int(ratio*height)]
#             annotation[object_nb]['center'] = [int(ratio*center_x), int(ratio*center_y + padding_y)]

#         return annotation

#     # def _resize_annotation_old(self, annotation, old_frame_shape, new_frame_shape):

#     #     old_shape_y, old_shape_x = old_frame_shape
#     #     new_shape_y, new_shape_x = new_frame_shape

#     #     for object_nb in annotation.keys():

#     #         [top_left_x, top_left_y, width, height] = annotation[object_nb]['bbox']
#     #         [center_x, center_y] = annotation[object_nb]['center']

#     #         annotation[object_nb]['bbox'] = [int(top_left_x * ratio_x), int(top_left_y * ratio_y), int(width * ratio_x), int(height * ratio_y)]
#     #         annotation[object_nb]['center'] = [int(center_x * ratio_x), int(center_y * ratio_y)]

#     #     return annotation

#     def __len__(self):
#         return len(self.video.num_frames)      
 