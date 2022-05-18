import os
from collections import defaultdict
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import multivariate_normal
from torch.utils.data import DataLoader
from skvideo.io import FFmpegWriter
from skimage.transform import downscale_local_mean

from plasticorigins.detection.transforms import TransformFrames
from plasticorigins.tools.video_readers import TorchIterableFromReader


class GaussianMixture:
    def __init__(self, means, covariance, weights):
        self.components = [
            multivariate_normal(mean=mean, cov=covariance) for mean in means
        ]
        self.weights = weights

    def pdf(self, x):
        result = 0
        for weight, component in zip(self.weights, self.components):
            result += weight * component.pdf(x)
        return result

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def cdf(self, x):
        result = 0
        for weight, component in zip(self.weights, self.components):
            result += weight * component.cdf(x)
        return result


def exp_and_normalise(lw):
    w = np.exp(lw - lw.max())
    return w / w.sum()


def in_frame(position, shape, border=0.02):
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


def gather_filenames_for_video_in_annotations(video, images, data_dir):
    images_for_video = [image for image in images if image["video_id"] == video["id"]]
    images_for_video = sorted(images_for_video, key=lambda image: image["frame_id"])

    return [os.path.join(data_dir, image["file_name"]) for image in images_for_video]


def get_detections_for_video(reader, detector, batch_size=16, device=None):
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


def overlay_transparent(background, overlay, x, y):
    """Overlays a transparent image over a background at topleft corner (x,y)"""
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
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype,)
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
    reader,
    output_detected,
    output_filename,
    skip_frames,
    maxframes,
    downscale,
    logger,
    gps_data=None,
    labels2icons=None,
):
    """Generates output video at 24 fps, with optional gps_data"""
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


def resize_external_detections(detections, ratio):
    for detection_nb in range(len(detections)):
        detection = detections[detection_nb]
        if len(detection):
            detection = np.array(detection)[:, :-1]
            detection[:, 0] = (detection[:, 0] + detection[:, 2]) / 2
            detection[:, 1] = (detection[:, 1] + detection[:, 3]) / 2
            detections[detection_nb] = detection[:, :2] / ratio
    return detections


def write_tracking_results_to_file(results, ratio_x, ratio_y, output_filename):
    """writes the output result of a tracking the following format:
    - frame
    - id
    - x_tl, y_tl, w=0, h=0
    - 4x unused=-1
    """
    with open(output_filename, "w") as output_file:
        for result in results:
            output_file.write(
                "{},{},{},{},{},{},{},{},{},{}\n".format(
                    result[0] + 1,
                    result[1] + 1,
                    round(ratio_x * result[2], 2),
                    round(ratio_y * result[3], 2),
                    0,  # width
                    0,  # height
                    round(result[4], 2),
                    result[5],
                    -1,
                    -1,
                )
            )


def read_tracking_results(input_file):
    """read the input filename and interpret it as tracklets
    i.e. lists of lists
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


# def gather_tracklets(tracklist):
#     """ Converts a list of flat tracklets into a list of lists
#     """
#     tracklets = defaultdict(list)
#     for track in tracklist:
#         frame_id = track[0]
#         track_id = track[1]
#         center_x = track[2]
#         center_y = track[3]
#         tracklets[track_id].append((frame_id, center_x, center_y))

#     tracklets = list(tracklets.values())
#     return tracklets


class FramesWithInfo:
    def __init__(self, frames, output_shape=None):
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
    """Display tracking"""

    def __init__(self, on, interactive=True):
        self.on = on
        self.fig, self.ax = plt.subplots()
        self.interactive = interactive
        if interactive:
            plt.ion()
        self.colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.legends = []
        self.plot_count = 0

    def display(self, trackers):

        something_to_show = False
        for tracker_nb, tracker in enumerate(trackers):
            if tracker.enabled:
                tracker.fill_display(self, tracker_nb)
                something_to_show = True

        self.ax.imshow(self.latest_frame_to_show)

        if len(self.latest_detections):
            self.ax.scatter(
                self.latest_detections[:, 0], self.latest_detections[:, 1], c="r", s=40,
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

    def update_detections_and_frame(self, latest_detections, frame):
        self.latest_detections = latest_detections
        self.latest_frame_to_show = cv2.cvtColor(
            cv2.resize(frame, self.display_shape), cv2.COLOR_BGR2RGB
        )
