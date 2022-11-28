"""The ``coco_utils`` submodule provides several classes and functions to convert coco objects.

This submodule allows the user to :

- filter and remap coco categories
- convert coco objects (polygons) to masks (torch tensors)
- convert coco objects (polygons) to bounding boxes
- check validated conversion

This submodule contains the following classes:

- ``CocoDetectionWithExif`` : Perform Coco detection with exif.
- ``ConvertCocoPolysToBboxes`` : Convert Coco Polygons to bounding boxes.
- ``ConvertCocoPolysToMask`` : Convert Coco Polygons to Masks.
- ``FilterAndRemapCocoCategories`` : Filter and remap Coco categories.
- ``GroupInTensor`` : Group arrays in (torch) Tensors.

This submodule contains the following functions:

- ``coco_remove_images_without_annotations(dataset:Dataset, cat_list:Optional[List[Dict]]=None)`` : Check if the annotation are valid or not empty.
- ``convert_coco_poly_to_mask(segmentations:List[Any], height:int, width:int)`` : Converting Coco polygons to torch masks (tensors).
- ``get_coco(root:str, image_set:str, transforms:Callable)`` : Transform images to a dataset with specified transformation.
- ``get_surfrider(root:str, image_set:str, transforms:Callable)`` : Transform images to a dataset with specified transformation.
- ``get_surfrider_video_frames(root:str, image_set:str, transforms:Callable)`` : Transform images to a dataset with specified transformation.

"""

import copy
import os
from typing import Any, Callable, Optional, Tuple, List, Dict
from numpy import ndarray
import imageio
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image
from pycocotools import mask as coco_mask

from .transforms import Compose


class FilterAndRemapCocoCategories:

    """Filter and remap Coco categories.

    Args:
        categories (List): list of object classes
        remap (bool): Remapping if ``True``, ``False`` if not
    """

    def __init__(self, categories: List, remap: bool = True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image: Image, anno: List[Dict]) -> Tuple[Any, List[Dict]]:

        """Remapping the images with true category ids.

        Args:
            image (Image): Image, from the instance file
            anno (List[Dict]): Annotations linked to the specified image, from instance file

        Returns:
            image (Image): Image, from the instance file
            anno (List[Dict]): Remapped annotations linked to the specified image, from instance file
        """

        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        return image, anno


def convert_coco_poly_to_mask(
    segmentations: List[Any], height: int, width: int
) -> torch.Tensor:

    """Converting Coco polygons to masks (torch tensors).

    Args:
        segmentations (List): list of polygons
        height (int): height of the mask
        width (int): width of the mask

    Returns:
        masks (torch.Tensor): output masks
    """

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


class ConvertCocoPolysToMask:

    """Convert Coco Polygons to Masks."""

    def __call__(self, image: Image, anno: List[Dict]) -> Tuple[Any, Any]:

        """Generating target image from masks.

        Args:
            image (Image): Image, from the instance file
            anno (List[Dict]): Annotations linked to the specified image, from instance file

        Returns:
            image (Image): Image, from the instance file
            target (Image): Target image resulting from the product between categories and masks
        """

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


class ConvertCocoPolysToBboxes:

    """Convert Coco Polygons to bounding boxes."""

    def __call__(self, image: Image, anno: List[Dict]) -> Tuple[Any, Dict]:

        """Building dictionnary of image annotations.

        Args:
            image (Image): Image, from the instance file
            anno (List[Dict]): Annotations linked to the specified image, from instance file

        Returns:
            image (Image): Image, from the instance file
            target (Dict): Target dictionnary with bounding boxes associated with object classes
        """

        bboxes = [obj["bbox"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]

        if bboxes:
            target = {"bboxes": bboxes, "cats": cats}

        else:
            target = {"bboxes": [], "cats": []}

        return image, target


class GroupInTensor:

    """Group arrays in (torch) Tensors."""

    def __call__(
        self, frames: ndarray, ground_truths: ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """Convert multi-dimensionnal numpy array frames and ground truths into torch tensors.

        Args:
            frames (ndarray): list of frames
            ground_truths (ndarray): the multi-dimensionnal ground truths array

        Returns:
            Both ``frames`` and ``ground_truths`` convert into torch tensors.
        """

        return torch.from_numpy(frames), torch.from_numpy(ground_truths)


def coco_remove_images_without_annotations(
    dataset: Dataset, cat_list: Optional[List[Dict]] = None
) -> Dataset:

    """Remove images without annotations.

    Args:
        dataset (Dataset): The image dataset to process.
        cat_list (Optional[List[Dict]]): The list of object categories. Set as default to ``None``.

    Returns:
        The filtered dataset.
    """

    def has_valid_annotation(anno: List[Dict]) -> bool:

        """Check if the annotations are valid or not empty.

        Args:
            anno (List[Dict]): Annotations linked to the specified image, from instance file

        Returns:
            ``True`` if the annotations exist and are valid.
            Otherwise, ``False``.
        """

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

        if has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)

    return dataset


def get_coco(root: str, image_set: str, transforms: Callable) -> Dataset:

    """Transform images to a dataset with specified transformation (classes ``FilterAndRemapCocoCategories``
        ``ConvertCocoPolysToMask`` and ``_coco_remove_images_without_annotations`` used).

    Args:
        root (str): Root directory where images are downloaded to
        image_set (str): directory to annotation data. Two possible values : ``"train"`` or ``"val"``.
        transforms (Callable): A function/transformation that takes input sample and its target as entry and returns a transformed version

    Returns:
        dataset (Dataset): dataset of images with specific transformations
    """

    PATHS = {
        "train": (
            "train2017",
            os.path.join("annotations", "instances_train2017.json"),
        ),
        "val": (
            "val2017",
            os.path.join("annotations", "instances_val2017.json"),
        ),
        # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    }
    CAT_LIST = [
        0,
        5,
        2,
        16,
        9,
        44,
        6,
        3,
        17,
        62,
        21,
        67,
        18,
        19,
        4,
        1,
        64,
        20,
        63,
        7,
        72,
    ]

    transforms = Compose(
        [
            FilterAndRemapCocoCategories(CAT_LIST, remap=True),
            ConvertCocoPolysToMask(),
            transforms,
        ]
    )

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = torchvision.datasets.CocoDetection(
        img_folder, ann_file, transforms=transforms
    )

    if image_set == "train":
        dataset = coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset


def get_surfrider(root: str, image_set: str, transforms: Callable) -> Dataset:

    """Transform images to a dataset with specified transformation (class ``CocoDetectionWithExif`` used).

    Args:
        root (str): Root directory where images are downloaded to
        image_set (str): directory to annotation data. Two possible values : ``"train"`` or ``"val"``.
        transforms (Callable): A function/transform that takes input sample and its target as entry and returns a transformed version.

    Returns:
        dataset (Dataset): dataset of images with specific transformations.
    """

    PATHS = {
        "train": (
            "Images_md5",
            os.path.join("annotations", "instances_train.json"),
        ),
        "val": (
            "Images_md5",
            os.path.join("annotations", "instances_val.json"),
        ),
        # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    }

    transforms = Compose(
        [
            # FilterAndRemapCocoCategories(CAT_LIST, remap=True),
            ConvertCocoPolysToMask(),
            # ConvertCocoPolysToBboxes(),
            transforms,
        ]
    )

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = torchvision.datasets.CocoDetection(
        img_folder, ann_file, transforms=transforms
    )

    # if image_set == "train":
    #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset


def get_surfrider_video_frames(
    root: str, image_set: str, transforms: Callable
) -> Dataset:

    """Transform images to a dataset with specified transformation (class `CocoDetection used`).

    Args:
        root (str): Root directory where images are downloaded to
        image_set (str): directory to annotation data. Two possible values : ``"train"`` or ``"val"``.
        transforms (Callable): A function/transform that takes input sample and its target as entry and returns a transformed version.

    Returns:
        dataset (Dataset): dataset of images with specific transformations
    """

    PATHS = {
        "train": (
            "images",
            os.path.join("annotations", "instances_train.json"),
        ),
        "val": ("images", os.path.join("annotations", "instances_val.json")),
        # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    }
    # CAT_LIST = [0, 1, 2, 3]

    transforms = Compose(
        [
            # FilterAndRemapCocoCategories(CAT_LIST, remap=True),
            # ConvertCocoPolysToMask(),
            ConvertCocoPolysToBboxes(),
            transforms,
        ]
    )

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetectionWithExif(img_folder, ann_file, transforms=transforms)

    # if image_set == "train":
    #     dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset


class CocoDetectionWithExif(torchvision.datasets.CocoDetection):

    """Perform Coco detection with exif.

    Args:
        root (str): path of root folder / data folder
        annFile (str): name of annotation file
        transform (Optional[Callable]): transformation for detection. Set as default to ``None``.
        target_transform (Optional[Callable]): target transformation for detection. Set as default to ``None``.
        transforms (Optional[Callable]): transformations for detection. Set as default to ``None``.
    """

    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root, annFile, transform, target_transform, transforms)

    def __getitem__(self, index: int) -> Tuple[ndarray, ndarray]:

        """Apply a transformation to the image which corresponds to the given index.

        Args:
            index (int): Index of item

        Returns:
            Tuple ``(image, target)`` with ``target`` is the object returned by ``coco.loadAnns``
        """

        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]["file_name"]
        img = imageio.imread(os.path.join(self.root, path))

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
