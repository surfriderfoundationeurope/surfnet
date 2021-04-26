import torchvision.transforms as torchvision_T
from . import transforms as T


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
    def __init__(self, base_size, crop_size, num_classes, downsampling_factor, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(2.0 * base_size)

        trans = [T.RandomResizeBboxes(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlipBboxes(hflip_prob))
        trans.extend([
            T.RandomCropBboxes(crop_size),
            T.ToTensorBboxes(num_classes, downsampling_factor),
            T.Normalize(mean=mean, std=std),
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
    def __init__(self, base_size, num_classes, downsampling_factor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResizeBboxes(base_size, base_size),
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


# class SegmentationPresetTrainPairs:
#     def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#         min_size = int(0.5 * base_size)
#         max_size = int(2.0 * base_size)

#         trans = [T.ToTensorNoCast(), T.RandomResize(min_size, max_size)]
#         if hflip_prob > 0:
#             trans.append(T.RandomHorizontalFlip(hflip_prob))
#         trans.extend([
#             T.RandomCrop(crop_size, stacked=True),
#             T.Normalize(mean=mean, std=std),
#             T.Displacement()
#         ])
#         self.transforms = T.Compose(trans)

#     def __call__(self, img, target):
#         return self.transforms(img, target)


# class SegmentationPresetEvalPairs:
#     def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
#         self.transforms = T.Compose([T.ToTensorNoCast(),
#             T.RandomResize(base_size, base_size),
#             T.Normalize(mean=mean, std=std),
#             T.Displacement()

#         ])

#     def __call__(self, img, target):
#         return self.transforms(img, target)