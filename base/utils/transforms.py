import numpy as np
from PIL import Image
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

from common.utils import blob_for_bbox

import math 
def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    padded=False
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
        padded=True
    return img, padded


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target

class RandomResizeBboxes(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        old_h, old_w = image.size
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        new_h, new_w = image.size
        ratio_h = new_h/old_h
        ratio_w = new_w/old_w
        for bbox_nb, bbox in enumerate(target['bboxes']):
            new_bbox = [ratio_w*bbox[0], ratio_h*bbox[1], ratio_w*bbox[2], ratio_h*bbox[3]]
            target['bboxes'][bbox_nb] = new_bbox

        return image, target



class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomHorizontalFlipBboxes(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            for bbox_nb, bbox in enumerate(target['bboxes']):
                new_bbox = bbox.copy()
                new_bbox[0] = image.size[0]-(new_bbox[0]+new_bbox[2])
                target['bboxes'][bbox_nb] = new_bbox
        return image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class RandomCropBboxes(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):

        image, padded = pad_if_smaller(image, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        new_h, new_w = image.size

        if not padded: 
            limit_top = crop_params[0]
            limit_bottom = crop_params[0] + crop_params[2]
            limit_left = crop_params[1]
            limit_right = crop_params[1] + crop_params[3]

            for bbox_nb, bbox in enumerate(target['bboxes']):

                top_left_x, top_left_y, width, height = bbox
                bottom_right_x, bottom_right_y = top_left_x + width, top_left_y + height

                # bbox_coords = np.array([[top_left_x, top_left_y],
                #                         [top_right_x, top_right_y],
                #                         [bottom_right_x, bottom_right_y],
                #                         [bottom_left_x, bottom_left_y]])

                # bbox_center_x, bbox_center_y = np.mean(bbox_coords,axis=0)

                if bottom_right_x < limit_left or bottom_right_y < limit_top or top_left_x > limit_right or top_left_y > limit_bottom:
                    del target['bboxes'][bbox_nb]
                    del target['cats'][bbox_nb]
                else:
                    new_top_left_x = max(0,top_left_x-limit_left)
                    new_top_left_y = max(0,top_left_y-limit_top)
                    new_bottom_right_x = min(new_w, bottom_right_x-limit_left)
                    new_bottom_right_y = min(new_h, bottom_right_y-limit_top)
                    new_bbox = [new_top_left_x, new_top_left_y, new_bottom_right_x-new_top_left_x, new_bottom_right_y-new_top_left_y]
                    target['bboxes'][bbox_nb] = new_bbox

            # target = F.crop(target, *crop_params)
        return image, target

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target

class ToTensor(object):

    def __call__(self, image, target):
        image = F.to_tensor(image)
        # target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class ToTensorBboxes(object):

    def __init__(self, num_classes, downsampling_factor):
        self.num_classes = num_classes
        self.downsampling_factor = downsampling_factor

    def __call__(self, image, target):
        h,w = image.size
        image = F.to_tensor(image)
        blobs = np.zeros(shape=(self.num_classes + 2, h // self.downsampling_factor, w // self.downsampling_factor))
        for i, bbox in enumerate(target['bboxes']):
            cat = target['cats'][i]-1
            blobs[cat], ct_int = blob_for_bbox(bbox,  blobs[cat], self.downsampling_factor)
            if ct_int is not None: 
                ct_x, ct_y = ct_int
                if ct_x < blobs.shape[2] and ct_y < blobs.shape[1]:
                    blobs[-2, ct_y, ct_x] = bbox[3]
                    blobs[-1, ct_y, ct_x] = bbox[2]
                # else:
                #     import matplotlib.pyplot as plt 
                #     print(ct_x, ct_y)
                #     fig, ax = plt.subplots(1,1,figsize=(20,20))
                #     ax.imshow(blobs[cat])
                #     import pickle
                #     with open('verbose.pickle','wb') as f: 
                #         pickle.dump((fig,ax),f)
                #     plt.close()
                

        target = torch.from_numpy(blobs)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target




class RandomResizeImageOnly(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        return image


    

# def blob_for_bbox(bbox, shape_x, shape_y, downsampling_factor):

#     # downsampling_factor = 4
#     [top_left_x, top_left_y, width, height] = [int(bbox_coord / downsampling_factor) for bbox_coord in bbox]

#     bbox_coords = [[top_left_x, top_left_y],
#                 [top_left_x+width, top_left_y],
#                 [top_left_x+width, top_left_y+height],
#                 [top_left_x, top_left_y+height]]
#     bbox_center = torch.from_numpy(np.mean(bbox_coords, axis=0))
#     sigma2 = min(width, height)
#     bbox_cov =  torch.from_numpy(np.diag([sigma2, sigma2]))

#     x = np.arange(shape_x // downsampling_factor)
#     y = np.arange(shape_y // downsampling_factor)
#     x2d, y2d = np.meshgrid(x, y)
#     pos = np.dstack((x2d, y2d))
#     rv = MultivariateNormal(loc=bbox_center,covariance_matrix=bbox_cov)
#     blob = torch.exp(rv.log_prob(torch.from_numpy(pos)))
#     # renormalization_coeff = np.sqrt(((2*np.pi)**2) * np.linalg.det(bbox_cov)) 
#     blob = blob / blob.max()
#     # blob[blob < 1e-16] = 1e-16
#     return blob


        


# class RandomHorizontalFlip(object):
#     def __init__(self, flip_prob):
#         self.flip_prob = flip_prob

#     def __call__(self, image, target):
#         if random.random() < self.flip_prob:
#             image = F.hflip(image)
#             target = F.hflip(target)
#         return image, target

# class ToTensorNoCast(object):

#     def __call__(self, image, target):
#         image = F.to_tensor(image)
#         target = torch.as_tensor(np.array(target))
#         return image, target

# class StackTensors(object):
#     def __call__(self, images, targets):
#         images = torch.stack([F.to_tensor(image) for image in images])
#         targets = torch.stack([torch.as_tensor(np.array(target), dtype=torch.int64) for target in targets])
#         return images, targets


# class Displacement(object):
#     def __init__(self, method='simple', multi_object=False):
#         self.method = method
#         self.multi_object = multi_object

#     def __call__(self, image, target_pair):
#         if self.method == 'simple':
#             if not self.multi_object:
#                 center_0 = torch.argmax(target_pair[0])
#                 center_1 = torch.argmax(target_pair[1])
#         return image, target_pair