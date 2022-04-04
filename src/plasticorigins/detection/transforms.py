import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import numpy as np
from plasticorigins.tools.misc import blob_for_bbox
import torch
ia.seed(1)
from torchvision.transforms import functional as F
import torchvision.transforms as T
import cv2

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensorBboxes(object):

    def __init__(self, num_classes, downsampling_factor):
        self.num_classes = num_classes
        self.downsampling_factor = downsampling_factor

    def __call__(self, image, bboxes):
        h,w = image.shape[:-1]
        image = F.to_tensor(image)
        if self.downsampling_factor is not None:
            blobs = np.zeros(shape=(self.num_classes + 2, h // self.downsampling_factor, w // self.downsampling_factor))
        else:
            blobs = np.zeros(shape=(self.num_classes + 2, h, w))

        for bbox_imgaug in bboxes:
            cat = bbox_imgaug.label-1
            bbox = [bbox_imgaug.x1, bbox_imgaug.y1, bbox_imgaug.width, bbox_imgaug.height]

            new_blobs, ct_int = blob_for_bbox(bbox,  blobs[cat], self.downsampling_factor)
            blobs[cat] = new_blobs
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

class TrainTransforms:
    def __init__(self, base_size, crop_size, num_classes, downsampling_factor, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.num_classes = num_classes
        self.downsampling_factor = downsampling_factor
        self.base_size = base_size
        self.crop_height, self.crop_width = crop_size
        self.hflip_prob = hflip_prob
        self.random_size_range = (int(self.base_size),int(2.0*self.base_size))
        self.seq = iaa.Sequential([
            iaa.Resize({"height": self.random_size_range, "width": "keep-aspect-ratio"}),
            iaa.Fliplr(p=self.hflip_prob),
            iaa.PadToFixedSize(width=self.crop_width, height=self.crop_height),
            iaa.CropToFixedSize(width=self.crop_width, height=self.crop_height)
        ])
        self.last_transforms = Compose([ToTensorBboxes(num_classes, downsampling_factor),
                                        Normalize(mean=mean,std=std)])



    def __call__(self, img, target):

        bboxes_imgaug = [BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[0]+bbox[2], y2=bbox[1]+bbox[3], label=cat) \
            for bbox, cat in zip(target['bboxes'],target['cats'])]
        bboxes = BoundingBoxesOnImage(bboxes_imgaug, shape=img.shape)

        img, bboxes_imgaug = self.seq(image=img, bounding_boxes=bboxes)
        return self.last_transforms(img, bboxes_imgaug)

class ValTransforms:
    def __init__(self, base_size, crop_size, num_classes, downsampling_factor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.num_classes = num_classes
        self.downsampling_factor = downsampling_factor
        self.base_size = base_size
        self.crop_height, self.crop_width = crop_size
        self.seq = iaa.Sequential([
            iaa.Resize({"height": int(self.base_size), "width": "keep-aspect-ratio"}),
            # iaa.Rotate((-45,45)),
            iaa.CenterPadToFixedSize(width=self.crop_width, height=self.crop_height),
            iaa.CenterCropToFixedSize(width=self.crop_width, height=self.crop_height)
        ])
        self.last_transforms = Compose([ToTensorBboxes(num_classes, downsampling_factor),
                                        Normalize(mean=mean,std=std)])



    def __call__(self, img, target):

        bboxes_imgaug = [BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[0]+bbox[2], y2=bbox[1]+bbox[3], label=cat) \
            for bbox, cat in zip(target['bboxes'],target['cats'])]
        bboxes = BoundingBoxesOnImage(bboxes_imgaug, shape=img.shape)

        img, bboxes_imgaug = self.seq(image=img, bounding_boxes=bboxes)
        return self.last_transforms(img, bboxes_imgaug)

class TransformFrames:
    def __init__(self):
        transforms = []

        transforms.append(T.Lambda(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]))

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)