"""The ``misc`` submodule provides computing functions for the prediction models.

This submodule contains the following class:

- ``ResizeForCenterNet`` : Resize images for Centernet model.

This submodule contains the following functions:

- ``blob_for_bbox(bbox:Union[List,array], heatmap:ndarray, downsampling_factor:Optional[float]=None)`` : Compute the 2-dimensional gaussian map (heat map) and the coordinates of the bounding box center.
- ``draw_umich_gaussian(heatmap:ndarray, center:Tuple[Union[float,int],Union[float,int]], radius:Union[float,int], k:int=1)`` : Draw the 2-dimensional gaussian map (heat map).
- ``gaussian2D(shape:Tuple[int,int], sigma:int=1)`` : Compute the 2-dimensional gaussian distribution.
- ``gaussian_radius(det_size:Tuple[int,int], min_overlap:float=0.7)`` : Compute the minimal gaussian radius.
- ``load_checkpoint(model:Any, trained_model_weights_filename:str)`` : Load the model checkpoint with specific model weights.
- ``load_model(arch:str, model_weights:Union[str,ndarray[Any,dtype[float64]]], device:str)`` : Load a prediction model with specific architecture, model weights and device type.

"""

import math
import numpy as np
from numpy import ndarray, array, dtype, float64
import torch
import torchvision.transforms.functional as F
from typing import Any, Tuple, Union, Optional, List

from plasticorigins.detection.centernet.models import create_model as create_base


class ResizeForCenterNet:

    """Resize images for Centernet model.

    Args:
        fix_res (bool): fixe resize for images with ``(H, W) = (512, 512).`` Set as default to ``False``.
    """

    def __init__(self, fix_res: bool = False):
        self.fix_res = fix_res

    def __call__(self, image: torch.Tensor) -> torch.Tensor:

        """Resize the input image according the fix_res for CenterNet model.

        Args:
            image (torch.Tensor): the input image to resize

        Returns:
            image (torch.Tensor): the resized image as output
        """

        if self.fix_res:
            new_h = 512
            new_w = 512

        else:
            w, h = image.size
            new_h = (h | 31) + 1
            new_w = (w | 31) + 1

        image = F.resize(image, (new_h, new_w))

        return image


def gaussian_radius(det_size: Tuple[int, int], min_overlap: float = 0.7) -> float:

    """Compute the minimal gaussian radius.

    Args:
        det_size (Tuple[int,int]): the size of the image / box (heigth, width)
        min_overlap (float): minimal overlap in [0,1]

    Returns:
        min_radius (float): The minimal gaussian radius
    """

    height, width = det_size

    # compute radius 1
    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    # compute radius 2
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    # compute radius 3
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape: Tuple[int, int], sigma: int = 1) -> ndarray:

    """Compute the 2-dimensional gaussian distribution.

    Args:
        shape (Tuple[int,int]): shape of the figure
        sigma (int): standard deviation. Set as default to 1.

    Returns:
        h (ndarray): 2-dimensional gaussian array
    """

    m, n = ((ss - 1.0) / 2.0 for ss in shape)
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0

    return h


def draw_umich_gaussian(
    heatmap: ndarray,
    center: Tuple[Union[float, int], Union[float, int]],
    radius: Union[float, int],
    k: int = 1,
) -> ndarray:

    """Draw the 2-dimensional gaussian heat map.

    Args:
        heatmap (ndarray): the input heat map
        center (Tuple[Union[float,int],Union[float,int]]): coordinates of the center of the 2D gaussian map
        radius (Union[float,int]): gaussian radius
        k (int): amplifier coefficient. Set as default to ``1``.

    Returns:
        masked_heatmap (ndarray) : the 2-dimensional gaussian heat map.
    """

    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def blob_for_bbox(
    bbox: Union[List, array],
    heatmap: ndarray,
    downsampling_factor: Optional[float] = None,
) -> Tuple[ndarray, array]:

    """Compute the 2-dimensional gaussian heat map and the coordinates of the bounding box center.

    Args:
        bbox (Union[List,array]): list of the bounding box coordinates
        heatmap (ndarray): the input heat map
        downsampling_factor (Optional[float]): downsampling factor. Set as default to ``None``.

    Returns:
        heatmap (ndarray): the 2-dimensional gaussian heat map
        ct_int (array): coordinates of the center
    """

    if downsampling_factor is not None:
        left, top, w, h = (bbox_coord // downsampling_factor for bbox_coord in bbox)

    else:
        left, top, w, h = (bbox_coord for bbox_coord in bbox)

    right, bottom = left + w, top + h
    ct_int = None

    if h > 0 and w > 0:
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.array([(left + right) / 2, (top + bottom) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        heatmap = draw_umich_gaussian(heatmap, ct_int, radius)

    return heatmap, ct_int


def load_checkpoint(model: Any, trained_model_weights_filename: str) -> Any:

    """Load the model checkpoint with specific model weights.

    Args:
        model (Any): prediction model
        trained_model_weights_filename (str): file name of the model weights

    Returns:
        model (Any): prediction model from specific checkpoint
    """

    checkpoint = torch.load(trained_model_weights_filename, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    return model


def load_model(
    arch: str,
    model_weights: Union[str, ndarray[Any, dtype[float64]]],
    device: str,
) -> Any:

    """Load a prediction model with specific architecture, model weights and device type.

    Args:
        arch (str): architecture of the model ("mobilenet_v3_small", "res_18", "dla_34")
        model_weights (Union[str,ndarray[Any,dtype[float64]]]): path of the prediction model weights or model weights
        device (str): type of device used ("cpu", "cuda", ...)

    Returns:
        model (Any): prediction model with specific architecture, model weights and device type from specific checkpoint
    """

    if model_weights is None:
        if arch == "mobilenet_v3_small":
            model_weights = "models/mobilenet_v3_pretrained.pth"
            arch = "mobilenetv3small"
        elif arch == "res_18":
            model_weights = "models/res18_pretrained.pth"
        elif arch == "dla_34":
            model_weights = "models/dla_34_pretrained.pth"

    heads = {"hm": 1} if arch != "dla_34" else {"hm": 1, "wh": 2}

    model = create_base(arch, heads=heads, head_conv=256).to(device)
    model = load_checkpoint(model, model_weights)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    return model
