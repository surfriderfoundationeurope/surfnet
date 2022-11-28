"""The ``detect`` submodule provides several detection functions for tracking application.

This submodule contains the following functions:

- ``detect(preprocessed_frames:Any, threshold:float, model:Any)`` : Return a detection Tensor with the localisation of the detected objects.
- ``nms(heat:torch.Tensor, kernel:int=3)`` : Applies a 2D max pooling over an input signal heat and compare the results before and after the max pooling.

"""

import torch
from numpy import ndarray
from typing import Any


def nms(heat: torch.Tensor, kernel: int = 3) -> torch.Tensor:

    """Applies a 2D max pooling over an input signal heat and compare the results before
        and after the max pooling.

    Args:
        heat (torch.Tensor): input heat map tensor (value in [0,1])
        kernel (int): kernel size for max pooling (filter kernel x kernel). Set as default to ``3`` (filter 3x3)

    Returns:
        The input heat map Tensor if the 2D max pooling is neutral. Else the null Tensor.
    """

    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def detect(preprocessed_frames: Any, threshold: float, model: Any) -> ndarray:

    """Return a detection Tensor with the localisation of the detected objects.

    Args:
        preprocessed_frames (Any): list of preprocessed frames from a DataLoader
        threshold (float): probability threshold for ground truths
        model (Any): Load a model pre-trained to apply to the preprocessed frames

    Returns:
        detections (ndarray): detection Tensor with the localisation of the detected objects
    """

    batch_result = torch.sigmoid(model(preprocessed_frames)["hm"])
    batch_peaks = nms(batch_result).gt(threshold).squeeze(dim=1)
    detections = [torch.nonzero(peaks).cpu().numpy()[:, ::-1] for peaks in batch_peaks]

    return detections
