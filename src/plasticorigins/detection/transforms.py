"""The ``transforms`` submodule provides several transformation classes to convert, process and normalize images.

This submodule contains the following classes:

- ``Compose`` : Compose class for creating transformation pipeline on images.
- ``Normalize`` : Apply normalization on images with specified mean and standard deviation.
- ``ToTensorBboxes`` : Transform tensors to bounding boxes.
- ``TrainTransforms`` : Apply transformations on images for training.
- ``TransformFrames`` : Transform frames (with normalization).
- ``ValTransforms`` : Apply transformations on images for validation.

"""

import cv2
import imgaug as ia
import numpy as np
from numpy import ndarray
from PIL import Image
import torch
import torchvision.transforms as T
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torchvision.transforms import functional as F
from typing import Any, Callable, Tuple, List, Dict, Union

from plasticorigins.tools.misc import blob_for_bbox

ia.seed(1)


class Compose:

    """Compose class for creating transformation pipeline on images.

    Args:
        transforms (Callable): transformations to apply
    """

    def __init__(self, transforms: Callable):
        self.transforms = transforms

    def __call__(self, image: Image, target: Any) -> Tuple[torch.Tensor, torch.Tensor]:

        """Apply transformations on image according to the given target.

        Args:
            image (Image): Image, from the instance file
            target (Any): target for transformations

        Returns:
            Tuple (image, target) after transformations
        """

        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class ToTensorBboxes:

    """Transform tensors to bounding boxes.

    Args:
        num_classes (int): number of different object classes
        downsampling_factor (float): downsampling factor for rescaling images
    """

    def __init__(self, num_classes: int, downsampling_factor: float):
        self.num_classes = num_classes
        self.downsampling_factor = downsampling_factor

    def __call__(
        self, image: Union[Any, ndarray], bboxes: List[Dict]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """Apply transformations on image according to the given target.

        Args:
            image (Union[Image,ndarray]): PIL Image, from the instance file or ndarray.
            bboxes (List[Dict]): List of annotations with labels and bounding boxes coordinates

        Returns:
            Tuple (image, target) after transformations
        """

        h, w = image.shape[:-1]
        image = F.to_tensor(image)

        if self.downsampling_factor is not None:
            blobs = np.zeros(
                shape=(
                    self.num_classes + 2,
                    h // self.downsampling_factor,
                    w // self.downsampling_factor,
                )
            )
        else:
            blobs = np.zeros(shape=(self.num_classes + 2, h, w))

        for bbox_imgaug in bboxes:
            cat = bbox_imgaug.label - 1
            bbox = [
                bbox_imgaug.x1,
                bbox_imgaug.y1,
                bbox_imgaug.width,
                bbox_imgaug.height,
            ]

            new_blobs, ct_int = blob_for_bbox(
                bbox, blobs[cat], self.downsampling_factor
            )

            blobs[cat] = new_blobs

            if ct_int is not None:
                ct_x, ct_y = ct_int

                if ct_x < blobs.shape[2] and ct_y < blobs.shape[1]:
                    blobs[-2, ct_y, ct_x] = bbox[3]
                    blobs[-1, ct_y, ct_x] = bbox[2]

        target = torch.from_numpy(blobs)

        return image, target


class Normalize:

    """Apply normalization on images with specified mean and standard deviation.

    Args:
        mean (List[float]): mean of normalization function
        std (List[float]): standard deviation of normalization function
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(
        self, image: torch.Tensor, target: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """Normalize the input image.

        Args:
            image (torch.Tensor): Image, from the instance file in torch Tensor format
            target (Any): target for normalization

        Returns:
            Tuple (image, target) after normalization
        """

        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, target


class TrainTransforms:

    """Apply transformations on images for training.

    Args:
        base_size (int): base size of images
        crop_size (Tuple[int,int]): crop size of images ``(H, W)``
        num_classes (int): number of object classes
        downsampling_factor (float): downsampling factor
        hflip_prob (float): High flip probability. Set as default to ``0.5``
        mean (Tuple[float,float,float]) : mean of normalization function. Set as default to ``(0.485, 0.456, 0.406)``
        std (Tuple[float,float,float]) : standard deviation of normalization function. Set as default to ``(0.229, 0.224, 0.225)``
    """

    def __init__(
        self,
        base_size: int,
        crop_size: Tuple[int, int],
        num_classes: int,
        downsampling_factor: float,
        hflip_prob: float = 0.5,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.num_classes = num_classes
        self.downsampling_factor = downsampling_factor
        self.base_size = base_size
        self.crop_height, self.crop_width = crop_size
        self.hflip_prob = hflip_prob
        self.random_size_range = (
            int(self.base_size),
            int(2.0 * self.base_size),
        )
        self.seq = iaa.Sequential(
            [
                iaa.Resize(
                    {
                        "height": self.random_size_range,
                        "width": "keep-aspect-ratio",
                    }
                ),
                iaa.Fliplr(p=self.hflip_prob),
                iaa.PadToFixedSize(width=self.crop_width, height=self.crop_height),
                iaa.CropToFixedSize(width=self.crop_width, height=self.crop_height),
            ]
        )
        self.last_transforms = Compose(
            [
                ToTensorBboxes(num_classes, downsampling_factor),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: ndarray, target: Any) -> Any:

        """Create a new augmenter sequence and apply transformations based on the argument ``last_transforms``.

        Args:
            image (ndarray): Image, from the instance file
            target (Any): target for transformations

        Returns:
            The last transformations required.
        """

        bboxes_imgaug = [
            BoundingBox(
                x1=bbox[0],
                y1=bbox[1],
                x2=bbox[0] + bbox[2],
                y2=bbox[1] + bbox[3],
                label=cat,
            )
            for bbox, cat in zip(target["bboxes"], target["cats"])
        ]
        bboxes = BoundingBoxesOnImage(bboxes_imgaug, shape=image.shape)

        image, bboxes_imgaug = self.seq(image=image, bounding_boxes=bboxes)

        return self.last_transforms(image, bboxes_imgaug)


class ValTransforms:

    """Apply transformations on images for validation.

    Args:
        base_size (Tuple[int,int]): base size of images ``(H, W)``
        crop_size (Tuple[int,int]): crop size of images ``(H, W)``
        num_classes (int): number of object classes
        downsampling_factor (float): downsampling factor
        mean (Tuple[float,float,float]) : mean of normalization function. Set as default to ``(0.485, 0.456, 0.406)``
        std (Tuple[float,float,float]) : standard deviation of normalization function. Set as default to ``(0.229, 0.224, 0.225)``
    """

    def __init__(
        self,
        base_size: Tuple[int, int],
        crop_size: Tuple[int, int],
        num_classes: int,
        downsampling_factor: float,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.num_classes = num_classes
        self.downsampling_factor = downsampling_factor
        self.base_size = base_size
        self.crop_height, self.crop_width = crop_size
        self.seq = iaa.Sequential(
            [
                iaa.Resize(
                    {
                        "height": int(self.base_size),
                        "width": "keep-aspect-ratio",
                    }
                ),
                # iaa.Rotate((-45,45)),
                iaa.CenterPadToFixedSize(
                    width=self.crop_width, height=self.crop_height
                ),
                iaa.CenterCropToFixedSize(
                    width=self.crop_width, height=self.crop_height
                ),
            ]
        )
        self.last_transforms = Compose(
            [
                ToTensorBboxes(num_classes, downsampling_factor),
                Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, image: ndarray, target: Dict) -> Any:

        """Create a new augmenter sequence and apply transformations based on the argument last_transforms.

        Args:
            image (ndarray): Image, from the instance file
            target (Dict): target for transformations (ground truths ``gt`` or heat map ``hm``)

        Returns:
            The last transformations required.
        """

        bboxes_imgaug = [
            BoundingBox(
                x1=bbox[0],
                y1=bbox[1],
                x2=bbox[0] + bbox[2],
                y2=bbox[1] + bbox[3],
                label=cat,
            )
            for bbox, cat in zip(target["bboxes"], target["cats"])
        ]
        bboxes = BoundingBoxesOnImage(bboxes_imgaug, shape=image.shape)

        image, bboxes_imgaug = self.seq(image=image, bounding_boxes=bboxes)

        return self.last_transforms(image, bboxes_imgaug)


class TransformFrames:

    """Transform frames (with normalization)."""

    def __init__(self):
        transforms = []

        transforms.append(T.Lambda(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        transforms.append(T.ToTensor())
        transforms.append(
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

        self.transforms = T.Compose(transforms)

    def __call__(self, image: ndarray) -> Any:

        """Apply given transformations on the input image.

        Args:
            image (ndarray): Image, from the instance file

        Returns:
            The image with the transformations required.
        """

        return self.transforms(image)
