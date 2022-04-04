import copy
import torch
import torch.utils.data
import torchvision
from PIL import Image, ExifTags
from typing import Callable, Optional, Tuple, Any

import os

from pycocotools import mask as coco_mask
from .transforms import Compose
import imageio 
class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, anno):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        return image, anno


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = torch.zeros((h, w), dtype=torch.uint8)
        target = Image.fromarray(target.numpy())
        return image, target

class ConvertCocoPolysToBboxes(object):
    def __call__(self, image, anno):
        bboxes = [obj["bbox"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if bboxes:
            target = {'bboxes':bboxes,'cats':cats}
        else:
            target = {'bboxes':[],'cats':[]}

        return image, target




class GroupInTensor(object):
    def __call__(self, frames, ground_truths):
        return torch.from_numpy(frames), torch.from_numpy(ground_truths)



def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def get_coco(root, image_set, transforms):
    PATHS = {
        "train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
        "val": ("val2017", os.path.join("annotations", "instances_val2017.json")),
        # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    }
    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]

    transforms = Compose([
        FilterAndRemapCocoCategories(CAT_LIST, remap=True),
        ConvertCocoPolysToMask(),
        transforms
    ])

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset

def get_surfrider_old(root, image_set, transforms):
    PATHS = {
        "train": ("Images_md5", os.path.join("annotations", "instances_train.json")),
        "val": ("Images_md5", os.path.join("annotations", "instances_val.json")),
        # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    }
    # CAT_LIST = [0, 1, 2, 3]

    transforms = Compose([
        # FilterAndRemapCocoCategories(CAT_LIST, remap=True),
        ConvertCocoPolysToMask(),
        # ConvertCocoPolysToBboxes(),
        transforms
    ])

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)

    # if image_set == "train":
    #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset

def get_surfrider(root, image_set, transforms):
    PATHS = {
        "train": ("images", os.path.join("annotations", "instances_train.json")),
        "val": ("images", os.path.join("annotations", "instances_val.json")),
        # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    }
    # CAT_LIST = [0, 1, 2, 3]

    transforms = Compose([
        # FilterAndRemapCocoCategories(CAT_LIST, remap=True),
        # ConvertCocoPolysToMask(),
        ConvertCocoPolysToBboxes(),
        transforms
    ])

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetectionWithExif(img_folder, ann_file, transforms=transforms)

    # if image_set == "train":
    #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset

def get_surfrider_video_frames(root, image_set, transforms):
    PATHS = {
        "train": ("data", os.path.join("annotations", "annotations_train.json")),
        "val": ("data", os.path.join("annotations", "annotations_val.json")),
        # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    }
    # CAT_LIST = [0, 1, 2, 3]

    transforms = Compose([
        ConvertCocoPolysToBboxes(),
        transforms
    ])

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = torchvision.datasets.CocoDetection(img_folder, ann_file, transforms=transforms)

    # if image_set == "train":
    #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset


class CocoDetectionWithExif(torchvision.datasets.CocoDetection):

    def __init__(
            self,
            root: str,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super(CocoDetectionWithExif, self).__init__(root, annFile, transform, target_transform, transforms)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']
        img = imageio.imread(os.path.join(self.root, path))

        # try:
        #     img = Image.open(os.path.join(self.root, path)).convert('RGB')
        #     for orientation in ExifTags.TAGS.keys():
        #         if ExifTags.TAGS[orientation]=='Orientation':
        #             break
            
        #     exif = img._getexif()
        #     if exif is not None:
        #         if exif[orientation] == 3:
        #             img=img.rotate(180, expand=True)
        #         elif exif[orientation] == 6:
        #             img=img.rotate(270, expand=True)
        #         elif exif[orientation] == 8:
        #             img=img.rotate(90, expand=True)

        # except (AttributeError, KeyError, IndexError):
        #     # cases: image don't have getexif
        #     pass
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
