from imgaug.augmenters.flip import Fliplr, VerticalFlip
from imgaug.augmenters.geometric import Rotate
from imgaug.augmenters.size import Resize
from matplotlib import image
from matplotlib.colors import Normalize
import torchvision.transforms as torchvision_T
from . import transforms as T
import matplotlib.pyplot as plt
import imgaug as ia
from imgaug import augmenters as iaa 
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import parameters as iap

ia.seed(1)
# plt.ioff()
class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        trans.extend([
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

class SegmentationPresetTrainBboxes:
    def __init__(self, base_size, crop_size, num_classes, downsampling_factor, hflip_prob=0.5, vflip_prob=0.2, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResizeBboxes(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlipBboxes(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlipBboxes(vflip_prob))
        trans.extend([
            T.RandomCropBboxes(crop_size),
            T.ToTensorBboxes(num_classes, downsampling_factor),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.Normalize(mean=mean, std=std)
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)
        
class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

class SegmentationPresetEvalBboxes:
    def __init__(self, base_size, crop_size, num_classes, downsampling_factor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResizeBboxes(base_size, base_size),
            T.CropBboxes(crop_size),
            T.ToTensorBboxes(num_classes, downsampling_factor),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)

class HeatmapExtractPreset:
    def __init__(self, mean=(0.408, 0.447, 0.47), std=(0.289, 0.274, 0.2785)):
        self.transform = torchvision_T.Compose([
            T.RandomResizeImageOnly(),
            torchvision_T.ToTensor(),
            torchvision_T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img):
        return self.transform(img)

class ImgAugPresetTrain:
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
        self.last_transforms = T.Compose([T.ToTensorBboxes(num_classes, downsampling_factor),
                                        T.Normalize(mean=mean,std=std)])


        
    def __call__(self, img, target):
        
        bboxes_imgaug = [BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[0]+bbox[2], y2=bbox[1]+bbox[3], label=cat) \
            for bbox, cat in zip(target['bboxes'],target['cats'])]
        bboxes = BoundingBoxesOnImage(bboxes_imgaug, shape=img.shape)

        img, bboxes_imgaug = self.seq(image=img, bounding_boxes=bboxes)
        # ia.imshow(bbs_aug.draw_on_image(image_aug, size=2))
        return self.last_transforms(img, bboxes_imgaug)

class ImgAugPresetVal:
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
        self.last_transforms = T.Compose([T.ToTensorBboxes(num_classes, downsampling_factor),
                                        T.Normalize(mean=mean,std=std)])


        
    def __call__(self, img, target):
        
        bboxes_imgaug = [BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[0]+bbox[2], y2=bbox[1]+bbox[3], label=cat) \
            for bbox, cat in zip(target['bboxes'],target['cats'])]
        bboxes = BoundingBoxesOnImage(bboxes_imgaug, shape=img.shape)

        img, bboxes_imgaug = self.seq(image=img, bounding_boxes=bboxes)
        # ia.imshow(bbs_aug.draw_on_image(image_aug, size=2))
        return self.last_transforms(img, bboxes_imgaug)