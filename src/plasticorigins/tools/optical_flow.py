"""The ``optical_flow`` submodule provides several functions for computing dense optical flow.

This submodule contains the following functions:

- ``compute_flow(frame0:Mat, frame1:Mat, downsampling_factor:Union[float,int])`` : Resizes the input frames and computes a dense optical flow using ``flow_opencv_dense``.
- ``flow_opencv_dense(img:Mat, img2:Mat)`` : Computes a dense optical flow using the Gunnar Farneback's algorithm.

"""

import cv2
from cv2 import Mat
from typing import Union


def flow_opencv_dense(img: Mat, img2: Mat) -> Mat:

    """Computes a dense optical flow using the Gunnar Farneback's algorithm.

    Args:
        img (Mat): first input image as a numpy matrix
        img2 (Mat): second input image of the same size and the same type as img, as a numpy matrix

    Returns:
        flow (Mat): the dense optical flow for both images
    """

    prvs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow


def compute_flow(
    frame0: Mat, frame1: Mat, downsampling_factor: Union[float, int]
) -> Mat:

    """Resizes the input frames and computes a dense optical flow using ``flow_opencv_dense``.

    Args:
        frame0 (Mat): first input frame as a matrix
        frame1 (Mat): second input frame of the same size and the same type as ``frame0``
        downsampling_factor (Union[float,int]) : downsampling factor to resize the frames

    Returns:
        The dense optical flow for both frames.
    """

    if downsampling_factor > 1:
        h, w = frame0.shape[:-1]
        new_h = h // downsampling_factor
        new_w = w // downsampling_factor

        frame0 = cv2.resize(frame0, (new_w, new_h))
        frame1 = cv2.resize(frame1, (new_w, new_h))

    return flow_opencv_dense(frame0, frame1)
