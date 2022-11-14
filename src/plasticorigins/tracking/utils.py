"""The ``utils`` submodule provides several useful functions for computes, annotations, detection and tracking.

This submodule contains the following class:

- ``Display`` : Display tracking.
- ``FramesWithInfo`` : Frames with information.
- ``GaussianMixture`` : This class contains several methods to compute gaussian Probability Density Function.

This submodule contains the following functions:

- ``exp_and_normalize(lw:ndarray[Any,dtype[float64]])`` : This fonction transforms a weight vector / matrix into a normalized gaussian vector / matrix.
- ``gather_filenames_for_video_in_annotations(video:Dict, images:List[Dict], data_dir:str)`` : Gather image names from a video for annotations.
- ``generate_video_with_annotations(reader:Any,output_detected:Dict,output_filename:str,skip_frames:int,downscale:float,
    logger:Logger,gps_data:Optional[ndarray]=None,labels2icons:Optional[ndarray]=None)`` : Generates output video at 24 fps, with optional ``gps_data``.
- ``get_detections_for_video(reader:Any, detector:Any, batch_size:int=16, device:Optional[str]=None)`` : Get detections for video.
- ``in_frame(position:Union[Tuple[int,int],List[int,int],array[int,int]],
                shape:Union[Tuple[int,int],List[int,int],array[int,int]],
                border:float=0.02)`` : Check if the (object) position is inside the image.
- ``overlay_transparent(background:ndarray, overlay:ndarray, x:int, y:int)`` : Overlays a transparent image over a background at topleft corner (x,y).
- ``read_tracking_results(input_file:str) `` : Read the input filename and interpret it as tracklets.
- ``resize_external_detections(detections:ndarray, ratio:float)`` : Resize external detections with specific ratio.
- ``write_tracking_results_to_file(results:ndarray, ratio_x:float, ratio_y:float, output_filename:str)`` : Writes the output results of a tracking

"""

import os
from collections import defaultdict
from time import time
from logging import Logger
from typing import Any, Iterable, Union, Tuple, List, Dict, Optional

import cv2
from cv2 import Mat
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, generic, ndarray, dtype, float64

import torch
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from skvideo.io import FFmpegWriter
from skimage.transform import downscale_local_mean

from plasticorigins.detection.transforms import TransformFrames
from plasticorigins.tools.video_readers import TorchIterableFromReader


class GaussianMixture:

    """This class contains several methods to compute gaussian Probability Density Function (pdf),
    the standard LOGistic Probability Density Function (logpdf) and the Cumulative Distribution Function (cdf).

    Args:
        means (ndarray[Any,dtype[float64]]]): the mean vector of the gaussian model
        covariance (ndarray[Any,dtype[float64]]]): the covariance matrix of the gaussian model
        weights (ndarray[Any,dtype[float64]]]): the weights of the predictive model
    """

    def __init__(
        self,
        means: ndarray[Any, dtype[float64]],
        covariance: ndarray[Any, dtype[float64]],
        weights: ndarray[Any, dtype[float64]],
    ):
        self.components = [
            multivariate_normal(mean=mean, cov=covariance) for mean in means
        ]
        self.weights = weights

    def pdf(self, x: ndarray) -> ndarray:

        """This method computes the Probability Density Function (pdf) for Gaussian Mixture.

        Args:
            x (ndarray[Any]): the input vector

        Returns:
            result (ndarray[Any,dtype[float64]]) : the Probability Density Function for the input vector x
        """

        result = 0
        for weight, component in zip(self.weights, self.components):
            result += weight * component.pdf(x)
        return result

    def logpdf(self, x: ndarray) -> ndarray:

        """This method computes the standard LOGistic (i.e, mean=0, std=π/sqrt(3)) Probability Density Function (logpdf) for Gaussian Mixture.

        Args:
            x (ndarray[Any]): the input vector

        Returns:
            result (ndarray[Any,dtype[float64]]) : the standard LOGistic Probability Density Function for the input vector x
        """

        return np.log(self.pdf(x))

    def cdf(self, x: ndarray) -> ndarray:

        """This method computes the Cumulative Density Function (pdf) for Gaussian Mixture.

        Args:
            x (ndarray[Any]): the input vector

        Returns:
            result (ndarray[Any,dtype[float64]]) : the Cumulative Density Function for the input vector x
        """

        result = 0
        for weight, component in zip(self.weights, self.components):
            result += weight * component.cdf(x)
        return result


def exp_and_normalize(lw: ndarray[Any, dtype[float64]]) -> ndarray[Any, dtype[float64]]:

    """This fonction transforms a weight vector / matrix into a normalized gaussian vector / matrix.

    Args:
        lw (ndarray[Any,dtype[float64]]): the input vector / matrix to normalize

    Returns:
        w (ndarray[Any,dtype[float64]]): the normalized gaussian weight vector / matrix
    """

    w = np.exp(lw - lw.max())

    return w / w.sum()


def in_frame(
    position: Union[Tuple[int, int], List[int], array],
    shape: Union[Tuple[int, int], List[int], array],
    border: float = 0.02,
) -> bool:

    """Check if the (object) position is inside the image.

    Args:
        position (Union[Tuple[int,int],List[int,int],array[int,int]]): the ``(x, y)`` position of the object
        shape (Union[Tuple[int,int],List[int,int],array[int,int]]): the frame shape ``(H, W)``
        border (float): the thickness of the image border. Set as default to ``0.02``.

    Returns:
        in_frame (bool): ``True`` if the object is in the frame, ``False`` if not

    """

    shape_x = shape[1]
    shape_y = shape[0]
    x = position[0]
    y = position[1]

    return (
        x > border * shape_x
        and x < (1 - border) * shape_x
        and y > border * shape_y
        and y < (1 - border) * shape_y
    )


def gather_filenames_for_video_in_annotations(
    video: Dict, images: List[Dict], data_dir: str
) -> List[str]:

    """Gather image names from a video for annotations.

    Args:
        video (Dict): the input video with its `id` key
        images (List[Dict]): the list of all images
        data_dir (str): the data path for images

    Returns:
        list_img_names (List[Any,type[str]]): the list of the image paths from data directory
    """

    images_for_video = [image for image in images if image["video_id"] == video["id"]]
    images_for_video = sorted(images_for_video, key=lambda image: image["frame_id"])

    return [os.path.join(data_dir, image["file_name"]) for image in images_for_video]


def get_detections_for_video(
    reader: Any,
    detector: Any,
    batch_size: int = 16,
    device: Optional[str] = None,
) -> List[array]:

    """Get detections for video.

    Args:
        reader (Any): the video reader used
        detector (Any): the detector for video
        batch_size (int): the batch size. Set as default to ``16``.
        device (Optional[str]): the device used ("cpu", "cuda",...). Set as default to ``None``.

    Returns:
        detections (List[array]): the list of built detections from the input video
    """

    detections = []
    dataset = TorchIterableFromReader(reader, TransformFrames())
    loader = DataLoader(dataset, batch_size=batch_size)
    average_times = []
    with torch.no_grad():
        for preprocessed_frames in loader:
            time0 = time()
            detections_for_frames = detector(preprocessed_frames.to(device))
            average_times.append(time() - time0)
            for detections_for_frame in detections_for_frames:
                if len(detections_for_frame):
                    detections.append(detections_for_frame)
                else:
                    detections.append(np.array([]))
    print(f"Frame-wise inference time: {batch_size/np.mean(average_times)} fps")
    return detections


def overlay_transparent(
    background: ndarray, overlay: ndarray, x: int, y: int
) -> ndarray[Any, dtype[float64]]:

    """Overlays a transparent image over a background at topleft corner (x,y).

    Args:
        background (ndarray): the input background
        overlay (ndarray): the input overlay
        x (int): the x-position of the topleft corner
        y (int): the y-position of the topleft corner

    Returns:
        background (ndarray[Any,dtype[float64]]): the transformed background
    """

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones(
                    (overlay.shape[0], overlay.shape[1], 1),
                    dtype=overlay.dtype,
                )
                * 255,
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y : y + h, x : x + w] = (1.0 - mask) * background[
        y : y + h, x : x + w
    ] + mask * overlay_image

    return background


def generate_video_with_annotations(
    reader: Any,
    output_detected: dict,
    output_filename: str,
    skip_frames: int,
    downscale: float,
    logger: Logger,
    gps_data: Optional[ndarray] = None,
    labels2icons: Optional[ndarray] = None,
) -> None:

    """Generates output video at 24 fps, with optional ``gps_data``.

    Args:
        reader (Any): the video reader used
        output_detected (Dict): the dictionnary of detected trashs
        output_filename (str): the name of the output file
        skip_frames (int): the frequence to skip frames
        downscale (float): the downscale factor
        logger (Logger): the logger to use
        gps_data (Optional[ndarray]): the gps data. Set as default to ``None``.
        labels2icons (Optional[ndarray]): the multi-dimensionnal array with labels to icons. Set as default to ``None``.
    """

    fps = 24
    logger.info("---Intepreting json")
    results = defaultdict(list)
    for trash in output_detected["detected_trash"]:
        for k, v in trash["frame_to_box"].items():
            frame_nb = int(k) - 1
            object_nb = trash["id"] + 1
            object_class = trash["label"]
            center_x = v[0]
            center_y = v[1]
            results[frame_nb * (skip_frames + 1)].append(
                (object_nb, center_x, center_y, object_class)
            )
            # append next skip_frames
            if str(frame_nb + 2) in trash["frame_to_box"]:
                next_trash = trash["frame_to_box"][str(frame_nb + 2)]
                next_x = next_trash[0]
                next_y = next_trash[1]
                for i in range(1, skip_frames + 1):
                    new_x = center_x + (next_x - center_x) * i / (skip_frames + 1)
                    new_y = center_y + (next_y - center_y) * i / (skip_frames + 1)
                    results[frame_nb * (skip_frames + 1) + i].append(
                        (object_nb, new_x, new_y, object_class)
                    )
    logger.info("---Writing video")

    writer = FFmpegWriter(
        filename=output_filename,
        outputdict={
            "-pix_fmt": "rgb24",
            "-r": "%.02f" % fps,
            "-vcodec": "libx264",
            "-b": "5000000",
        },
    )

    font = cv2.FONT_HERSHEY_TRIPLEX
    for frame_nb, frame in enumerate(reader):
        detections_for_frame = results[frame_nb]
        for detection in detections_for_frame:
            if labels2icons is None:
                # write name of class
                cv2.putText(
                    frame,
                    f"{detection[0]}/{detection[3]}",
                    (int(detection[1]), int(detection[2]) + 5),
                    font,
                    2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
            else:
                # icons
                overlay_transparent(
                    frame,
                    labels2icons[detection[3]],
                    int(detection[1]) + 5,
                    int(detection[2]),
                )
                cv2.putText(
                    frame,
                    f"{detection[0]}",
                    (int(detection[1] + 46 + 5), int(detection[2]) + 42),
                    font,
                    1.2,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            if gps_data is not None:
                latitude = gps_data[frame_nb // fps]["Latitude"]
                longitude = gps_data[frame_nb // fps]["Longitude"]
                cv2.putText(
                    frame,
                    f"GPS:{latitude},{longitude}",
                    (10, frame.shape[0] - 30),
                    font,
                    2,
                    (0, 255, 0),
                    3,
                    cv2.LINE_AA,
                )

        frame = downscale_local_mean(frame, (downscale, downscale, 1)).astype(np.uint8)
        writer.writeFrame(frame[:, :, ::-1])

    writer.close()
    reader.video.release()

    logger.info("---finished writing video")


def resize_external_detections(detections: ndarray, ratio: float) -> ndarray:

    """Resize external detections with specific ratio.

    Args:
        detections (ndarray): the n-dimensionnal array of detections
        ratio (float): the ratio to resize detections

    Returns:
        detections (ndarray): the resized external detections
    """

    for detection_nb in range(len(detections)):
        detection = detections[detection_nb]
        if len(detection):
            detection = np.array(detection)[:, :-1]
            detection[:, 0] = (detection[:, 0] + detection[:, 2]) / 2
            detection[:, 1] = (detection[:, 1] + detection[:, 3]) / 2
            detections[detection_nb] = detection[:, :2] / ratio
    return detections


def write_tracking_results_to_file(
    results: ndarray, coord_mapping, output_filename: str
) -> None:

    """Writes the output results of a tracking in the following format:

    - frame
    - id
    - x_tl, y_tl, w=0, h=0
    - 4x unused=-1

    Args:
        results (ndarray): the output results
        coord_mapping (function): the mapping function for coordonates with ratio scaling
        output_filename (str): the name of the output file

    """

    with open(output_filename, "w") as output_file:
        for result in results:
            x, y = coord_mapping(result[2], result[3])
            output_file.write(
                "{},{},{},{},{},{},{},{},{},{}\n".format(
                    result[0] + 1,
                    result[1] + 1,
                    x,
                    y,
                    0,  # width
                    0,  # height
                    round(result[4], 2),
                    result[5],
                    -1,
                    -1,
                )
            )


def read_tracking_results(input_file: str) -> List[List]:

    """Read the input filename and interpret it as tracklets in this format : ``list[list,...]``.

    Args:
        input_file (str): the input file to read tracking results

    Returns:
        tracklets (List[List]): the tracking results stored in tracklets
    """

    raw_results = np.loadtxt(input_file, delimiter=",")
    if raw_results.ndim == 1:
        raw_results = np.expand_dims(raw_results, axis=0)
    tracklets = defaultdict(list)
    for result in raw_results:
        # Skip blank lines
        if result is None or len(result) == 0:
            continue
        frame_id = int(result[0])
        track_id = int(result[1])
        left, top, width, height = result[2:6]
        center_x = left + width / 2
        center_y = top + height / 2
        conf = result[6]
        class_id = int(result[7])
        tracklets[track_id].append((frame_id, center_x, center_y, conf, class_id))

    tracklets = list(tracklets.values())
    return tracklets


class FramesWithInfo:

    """Frames with information.

    Args:
        frames (ndarray): the list of frames
        output_shape (Optional[ndarray]): the shape of output frames
    """

    def __init__(self, frames: ndarray, output_shape: Optional[ndarray] = None):
        self.frames = frames
        if output_shape is None:
            self.output_shape = frames[0].shape[:-1][::-1]
        else:
            self.output_shape = output_shape
        self.end = len(frames)
        self.read_head = 0

    def __next__(self):
        if self.read_head < self.end:
            frame = self.frames[self.read_head]
            self.read_head += 1
            return frame
        else:
            raise StopIteration

    def __iter__(self):
        return self


class Display:

    """Display tracking.

    Args:
        on (bool): to activate display
        interactive (bool): to have interactive display. Set as default to ``True``.
    """

    def __init__(self, on: bool, interactive: bool = True):
        self.on = on
        self.fig, self.ax = plt.subplots()
        self.interactive = interactive
        if interactive:
            plt.ion()
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.legends = []
        self.plot_count = 0

    def display(self, trackers: Iterable) -> None:

        """To display trackers.

        Args:
            trackers (Iterable): the input trackers to display
        """

        something_to_show = False
        for tracker_nb, tracker in enumerate(trackers):
            if tracker.enabled:
                tracker.fill_display(self, tracker_nb)
                something_to_show = True

        self.ax.imshow(self.latest_frame_to_show)

        if len(self.latest_detections):
            self.ax.scatter(
                self.latest_detections[:, 0],
                self.latest_detections[:, 1],
                c="r",
                s=40,
            )

        if something_to_show:
            self.ax.xaxis.tick_top()
            plt.legend(handles=self.legends)
            self.fig.canvas.draw()
            if self.interactive:
                plt.show()
                while not plt.waitforbuttonpress():
                    continue
            else:
                plt.savefig(os.path.join("plots", str(self.plot_count)))
            self.ax.cla()
            self.legends = []
            self.plot_count += 1

    def update_detections_and_frame(
        self, latest_detections: ndarray[Any, dtype[generic]], frame: Mat
    ) -> None:

        """Update detections and frame.

        Args:
            latest_detections (ndarray): the lastest detections
            frame (Mat): the frame to update for resize
        """

        self.latest_detections = latest_detections
        self.latest_frame_to_show = cv2.cvtColor(
            cv2.resize(frame, self.display_shape), cv2.COLOR_BGR2RGB
        )
