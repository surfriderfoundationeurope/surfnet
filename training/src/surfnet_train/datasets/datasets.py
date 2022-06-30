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

    def __init__(self, annotations_dir, data_dir, split='val'):
        self.split = split
        if self.split == 'train':
            annotation_filename = os.path.join(annotations_dir,'annotations_train.json')
        else: 
            annotation_filename = os.path.join(annotations_dir,'annotations_val.json')

        with open(annotation_filename, 'r') as f: 
            self.annotations = json.load(f)

        self.data_dir = data_dir

        with open(os.path.join(data_dir,'shapes.pickle'),'rb') as f:
            (self.old_shape, self.new_shape) = pickle.load(f)

        self.flows_dir = os.path.join(data_dir,'flows')
        self.heatmaps_dir = os.path.join(data_dir,'heatmaps')

        self.subset = False

        self.init_ids()


    def __getitem__(self, index):

        location_id = self.ids[index]

        if self.split == 'train':
            heatmap0, gt0, gt1, flow01 = self.get_training_item(location_id)

            heatmap0 = heatmap0.unsqueeze(0)
            gt0 = torch.from_numpy(gt0).unsqueeze(0)
            gt1 = torch.from_numpy(gt1).unsqueeze(0)
            flow01 = -torch.from_numpy(flow01)

            return heatmap0, gt0, gt1, flow01
        else: 
            heatmap, gt = self.get_testing_item(location_id)

            heatmap = heatmap.unsqueeze(0)
            gt = torch.from_numpy(gt).unsqueeze(0)

            return heatmap, gt

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
        heatmap_filename = os.path.join(self.heatmaps_dir, image0['file_name'].split('/')[0], '{:03d}.pickle'.format(frame_id_0))
        with open(heatmap_filename,'rb') as f:
            heatmap0 = pickle.load(f)

        flow_filename = os.path.join(self.flows_dir, image0['file_name'].split('/')[0], '{:03d}_{:03d}.npy'.format(frame_id_0,frame_id_1))
        flow01 = np.load(flow_filename)
        bboxes0 = [annotation['bbox'] for annotation in annotations0]
        bboxes1 = [annotation['bbox'] for annotation in annotations1]
        gt0 = self.build_gt(bboxes0)
        gt1 = self.build_gt(bboxes1)

        return heatmap0, gt0, gt1, flow01 
    
    def get_testing_item(self, location_id):

        image, annotations = self.annotations[location_id[0]][location_id[1]]
        frame_id = image['frame_id']
        heatmap_filename = os.path.join(self.heatmaps_dir, image['file_name'].split('/')[0], '{:03d}.pickle'.format(frame_id))
        with open(heatmap_filename,'rb') as f:
            heatmap = pickle.load(f)

        bboxes = [annotation['bbox'] for annotation in annotations]
        gt = self.build_gt(bboxes)
        return heatmap, gt
    

            
    def build_gt(self, bboxes):

        gt = np.zeros(shape=self.new_shape)
        ratio_x = self.new_shape[1]/self.old_shape[1]
        ratio_y = self.new_shape[0]/self.old_shape[0]

        for bbox in bboxes:
            bbox = [ratio_x*bbox[0],ratio_y*bbox[1],ratio_x*bbox[2],ratio_y*bbox[3]]
            news_blobs, _ = blob_for_bbox(bbox, gt)
            gt = np.maximum(gt, news_blobs) 

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
 