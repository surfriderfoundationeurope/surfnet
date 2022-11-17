"""The ``video_readers`` submodule provides several classes for reading videos and frames.

This submodule contains the following classes:

- ``AdvancedFrameReader`` : Advanced reader for frames with specific parameters.
- ``IterableFrameReader`` : Iterable reader for frames.
- ``SimpleVideoReader`` : Simple reader for frames.
- ``TorchIterableFromReader`` : Torch Iterable reader for frames.

"""

import cv2
from cv2 import Mat
import torch
from tqdm import tqdm
from numpy import ndarray
from typing import Callable, List, Tuple, Union, Optional


def square_crop(input_frame: Mat, out_shape: Tuple[int, int]):

    """Crops the largest square in the center of the image.

    Args:
        input_frame (Mat): input image in ``cv2 Mat`` format.
        out_shape (Tuple[int, int]): the output shape of the resized image.

    Returns:
        The resized image with ``out_shape`` shape.
    """

    h, w, _ = input_frame.shape

    if h > w:
        new_h = w
        xtop = h // 2 - new_h // 2
        crop = input_frame[xtop : xtop + new_h, :, :]

    elif h < w:
        new_w = h
        yleft = w // 2 - new_w // 2
        crop = input_frame[:, yleft : yleft + new_w, :]

    else:
        crop = input_frame

    return cv2.resize(crop, out_shape)


class AdvancedFrameReader:

    """Advanced reader for frames with specific parameters.

    Args:
        video_name (str): name of the video / path to the video
        read_every (int): parameter set to ``1`` for reading the whole video (all the frames). ``0`` means "to skip".
        rescale_factor (Union[int,float]): rescale factor for frames
        init_time_min (int): number of minutes of the initial time
        init_time_s (int): number of seconds of the initial time
    """

    def __init__(
        self,
        video_name: str,
        read_every: int,
        rescale_factor: Union[int, float],
        init_time_min: int,
        init_time_s: int,
    ):
        self.cap = cv2.VideoCapture(video_name)

        self.original_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.original_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if rescale_factor != 1:
            self.set_rescale_factor(rescale_factor)
        else:
            self.original_shape_mode = True

        self.init_rescale_factor = rescale_factor

        self.frame_skip = read_every - 1
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) / read_every
        print(f"Reading at {self.fps:.2f} fps")

        self.set_time_position(init_time_min, init_time_s)
        self.init_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.total_num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def post_process(self, ret: bool, frame: Mat) -> Tuple[bool, Mat]:

        """Apply a post resize process on the read image if necessary (according the ret value).

        Args:
            ret (bool): if the ``read()`` function reads successfully the capture, ``ret = True``. Else, ``ret = False``.
            frame (Mat): frame to post process (resize)

        Returns:
            ret (bool): the same `ret` value as the input
            resized_frame (Mat): the resized frame or the original frame. If `ret` is ``False``, returns empty matrix.
        """

        if ret:
            if self.original_shape_mode:
                return ret, frame
            else:
                return ret, cv2.resize(frame, self.new_shape)
        else:
            return ret, []

    def skip(self) -> None:

        """Skip if there are no frames to read."""

        if not self.nb_frames_read:
            return
        else:
            for _ in range(self.frame_skip):
                self.cap.read()

    def read_frame(self) -> Tuple[bool, Mat]:

        """Read a frame.

        Returns:
            ret (bool): First output of the ``post_process`` method : if the ``read()`` function reads successfully the frame, ``ret = True``. Else, ``ret = False``.
            resized_frame (Mat): Second output of the ``post_process`` method : the resized frame or the original frame. If `ret` is False, returns empty matrix.
        """

        self.skip()
        ret, frame = self.cap.read()
        self.nb_frames_read += 1

        return self.post_process(ret, frame)

    def set_time_position(self, time_min: int, time_s: int) -> None:

        """Set time positions in seconds.

        Args:
            time_min (int): the number of minutes for the time position
            time_s (int): the number of seconds for the time position
        """

        time = 60 * time_min + time_s
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * time)
        self.init_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        # print('Reading from {}min{}sec'.format(time_min,time_s))
        self.nb_frames_read = 0

    def reset_init_frame(self) -> None:

        """Reset to the initial frame."""

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.init_frame))
        self.nb_frames_read = 0

    def set_init_frame(self, init_frame: Union[int, str]) -> None:

        """Set and start at the initial frame.

        Args:
            init_frame (Union[int,str]): the number of the initial frame
        """

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(init_frame))
        self.init_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.nb_frames_read = 0

    def set_rescale_factor(self, rescale_factor: Union[int, float]) -> None:

        """Rescale the size of a frame (width, height).

        Args:
            rescale_factor (Union[int,float]): the rescale factor
        """

        width = int(self.original_width / rescale_factor)
        height = int(self.original_height / rescale_factor)
        self.new_shape = (width, height)
        self.original_shape_mode = False
        # print('Reading in {}x{}'.format(width, height))

    def set_original_shape_mode(self, mode: bool) -> None:

        """Set the original shape mode.

        Args:
            mode (bool): the original shape mode is kept if ``mode = True``
        """

        self.original_shape_mode = mode

    def reset_init_rescale_factor(self) -> None:

        """Reset the initial rescale factor."""

        self.set_rescale_factor(self.init_rescale_factor)


class IterableFrameReader:

    """Iterable reader for frames.

    Args:
        video_filename (str): name of the video / path to the video
        skip_frames (int): parameter for skipping frames. Set as default to``0``. It means "read every".
        output_shape (Optional[Tuple[int,int]]): the shape of the outputs (frames). Set as default to ``None``.
        progress_bar (bool): to display the progress bar. Set as default to ``False``.
        preload (bool): If we want to preload all the frames. Set as default to ``False``,
        max_frame (int): the maximum number of frames to read. Set as default to ``0``.
    """

    def __init__(
        self,
        video_filename: str,
        skip_frames: int = 0,
        output_shape: Optional[Tuple[int, int]] = None,
        progress_bar: bool = False,
        preload: bool = False,
        max_frame: int = 0,
        crop: Optional[ndarray] = None,
    ):
        # store arguments for reset
        self.video_filename = video_filename
        self.max_frame_arg = max_frame
        self.progress_bar_arg = progress_bar
        self.preload = preload
        self.skip_frames = skip_frames
        self.crop = crop

        self.video = cv2.VideoCapture(video_filename)
        self.input_shape = (
            int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.total_num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.max_num_frames = (
            min(max_frame, self.total_num_frames)
            if max_frame != 0
            else self.total_num_frames
        )
        self.num_skipped_frames = 0
        self.counter = 0
        self.progress_bar = None

        if output_shape is None:
            w, h = self.input_shape
            new_h = (h | 31) + 1
            new_w = (w | 31) + 1
            self.output_shape = (new_w, new_h)
        else:
            self.output_shape = output_shape

        self.fps = self.video.get(cv2.CAP_PROP_FPS) / (self.skip_frames + 1)

        if self.preload:
            self.frames = self._load_all_frames()

    def update_progress_bar(self) -> None:

        """Update the bar progression about the the reading of the frames."""

        if self.progress_bar_arg:

            if self.progress_bar:
                # update_progress_bar
                self.progress_bar.update()

            else:
                # create progress bar
                self.progress_bar = tqdm(
                    total=int(self.max_num_frames / (self.skip_frames + 1)),
                    position=1,
                    leave=True,
                )

    def reset_video(self) -> None:

        """Reset the video. This method is needed as ``cv2.CAP_PROP_POS_FRAMES``
        does not work on all backends.
        """

        self.video.release()
        if self.progress_bar:
            self.progress_bar.close()
        self.__init__(
            self.video_filename,
            self.skip_frames,
            self.output_shape,
            self.progress_bar_arg,
            self.preload,
            self.max_frame_arg,
            self.crop,
        )

    def _load_all_frames(self) -> List[Mat]:

        """Load all frames after their reading.

        Returns:
            frames (List[Mat]): list of the frames
        """

        frames = []

        while True:
            ret, frame = self._read_frame()
            if ret:
                frames.append(frame)
            else:
                break

        if self.progress_bar:
            self.progress_bar.reset()

        return frames

    def __next__(self) -> Union[Mat, StopIteration]:

        """Attempt of reading a frame.

        Returns:
            The read frame or a stop iteration status
        """

        self.counter += 1

        if self.preload:
            if self.counter < len(self.frames):
                frame = self.frames[self.counter]
                self.update_progress_bar()
                return frame

        else:
            if self.counter < self.max_num_frames:
                ret, frame = self._read_frame()
                if ret:
                    return frame
                else:
                    self.num_skipped_frames += 1
                    return next(self)

        self.reset_video()

        raise StopIteration

    def _read_frame(self) -> Tuple[bool, Mat]:

        """Read a frame.

        Returns:
            ret (bool): If the ``read()`` function reads successfully the video, ``ret = True``. Else, ``ret = False``.
            frame (Mat): The resized frame or the original frame if not skipped
        """

        ret, frame = self.video.read()
        self._skip_frames()

        if ret:
            self.update_progress_bar()

            if self.crop:
                frame = square_crop(frame, self.output_shape)

            else:
                frame = cv2.resize(frame, self.output_shape)

        return ret, frame

    def __iter__(self) -> None:
        return self

    def _skip_frames(self) -> None:

        """Skip frames."""

        for _ in range(self.skip_frames):
            self.counter += 1
            self.video.read()

    def get_inv_mapping(self, downsampling_factor: float):

        """Returns a mapping between coordinates in cropped space
        and coordinates in the original video"""

        h_in, w_in = self.input_shape
        x_top, y_left = 0, 0

        if self.crop:
            if h_in > w_in:
                h_new, w_new = w_in, w_in
                x_top = h_in // 2 - h_new // 2
                y_left = 0
            elif h_in < w_in:
                h_new, w_new = h_in, h_in
                y_left = w_in // 2 - w_new // 2
                x_top = 0
            else:
                h_new, w_new = w_in, w_in
        else:
            h_new, w_new = h_in, w_in

        ratio_x, ratio_y = (
            h_new * downsampling_factor / self.output_shape[0],
            w_new * downsampling_factor / self.output_shape[1],
        )

        mapping = lambda x, y: (int(x * ratio_x + x_top), int(y * ratio_y + y_left))
        return mapping


class SimpleVideoReader:

    """Simple reader for frames.

    Args:
        video_filename (str): name of the video / path to the video
        skip_frames (int): parameter for skipping frames. Set as default to``0``. It means "read every".
    """

    def __init__(self, video_filename: str, skip_frames: int = 0):
        self.skip_frames = skip_frames
        self.video = cv2.VideoCapture(video_filename)
        self.shape = (
            int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        self.fps = self.video.get(cv2.CAP_PROP_FPS) / (skip_frames + 1)
        self.frame_nb = 0
        self.num_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)

    def read(self) -> Tuple[bool, Mat, int]:

        """Reads a frame and increments the `frame_nb' of read frames.

        Returns:
            ret (bool): If the ``read()`` function reads successfully the video, ``ret = True``. Else, ``ret = False``.
            frame (Mat): the read frame
            frame_nb (int): the number of already read frames
        """

        ret, frame = self.video.read()
        self.frame_nb += 1
        self._skip_frames()

        return ret, frame, self.frame_nb - 1

    def set_frame(self, frame_nb_to_set: int) -> None:

        """Set ``frame_nb_to_set`` frames.

        Args:
            frame_nb_to_set (int): the number of frames to set
        """

        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_nb_to_set)
        self.frame_nb = frame_nb_to_set

    def _skip_frames(self) -> None:

        """Skip frames."""

        for _ in range(self.skip_frames):
            self.video.read()


class TorchIterableFromReader(torch.utils.data.IterableDataset):

    """Torch Iterable reader for frames.

    Args:
        reader (torch.utils.data.IterableDataset): the dataset of read images
        transforms (Callable): the specific transformations for frames
    """

    def __init__(self, reader: torch.utils.data.IterableDataset, transforms: Callable):
        self.transforms = transforms
        self.reader = reader

    def __iter__(self) -> None:

        """Iterable transformations on frames."""

        for frame in self.reader:
            if frame is not None:
                yield self.transforms(frame)
