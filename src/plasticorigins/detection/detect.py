"""Provide several detection functions for tracking application.

The module contains the following functions:
- nms(heat:torch.Tensor, kernel:int=3) : Applies a 2D max pooling over an input signal heat and compare the results before and after the max pooling.
- detect(preprocessed_frames:Any, threshold:float, model:Any)

"""

from typing import Any
from numpy import ndarray
import torch


def nms(heat:torch.Tensor, kernel:int=3) -> torch.Tensor :

    """ Applies a 2D max pooling over an input signal heat and compare the results before 
        and after the max pooling. 

    Args:
        heat (Tensor): input tensor
        kernel (int): kernel size for max pooling (filter kernel x kernel). Set as default to 3 (filter 3x3).

    Returns:
        Tensor : The input Tensor heat if the 2D max pooling is neutral. Else the null Tensor.
    """

    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()

    return heat * keep


def detect(preprocessed_frames:Any, threshold:float, model:Any) -> ndarray:

    """ 
    Args:
        preprocessed_frames (Any): list of preprocessed frames from a DataLoader.
        threshold (float):
        model (Any): Load a model pre-trained to apply to the preprocessed frames.

    Returns:
        detections (ndarray): 
    """

    batch_result = torch.sigmoid(model(preprocessed_frames)["hm"])
    batch_peaks = nms(batch_result).gt(threshold).squeeze(dim=1)
    detections = [torch.nonzero(peaks).cpu().numpy()[:, ::-1] for peaks in batch_peaks]

    return detections
