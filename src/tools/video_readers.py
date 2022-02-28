import cv2
import torch
from tqdm import tqdm
from itertools import cycle


class AdvancedFrameReader:
    def __init__(self, video_name, read_every, rescale_factor, init_time_min, init_time_s):

        self.cap = cv2.VideoCapture(video_name)

        self.original_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.original_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if rescale_factor != 1:
            self.set_rescale_factor(rescale_factor)
        else:
            self.original_shape_mode = True

        self.init_rescale_factor = rescale_factor

        self.frame_skip = read_every -  1
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)/read_every
        print(f'Reading at {self.fps:.2f} fps')

        self.set_time_position(init_time_min, init_time_s)
        self.init_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.total_num_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def post_process(self, ret, frame):
        if ret:
            if self.original_shape_mode:
                return ret, frame
            else:
                return ret, cv2.resize(frame, self.new_shape)
        else:
            return ret, []

    def skip(self):
        if not self.nb_frames_read:
            return
        else:
            for _ in range(self.frame_skip):
                self.cap.read()

    def read_frame(self):

        self.skip()
        ret, frame = self.cap.read()
        self.nb_frames_read += 1
        return self.post_process(ret, frame)

    def set_time_position(self, time_min, time_s):
        time = 60 * time_min  + time_s
        self.cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * time)
        self.init_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        #print('Reading from {}min{}sec'.format(time_min,time_s))
        self.nb_frames_read = 0

    def reset_init_frame(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,int(self.init_frame))
        self.nb_frames_read = 0

    def set_init_frame(self, init_frame):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(init_frame))
        self.init_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.nb_frames_read = 0

    def set_rescale_factor(self, rescale_factor):
        width = int(self.original_width/rescale_factor)
        height = int(self.original_height/rescale_factor)
        self.new_shape  = (width, height)
        self.original_shape_mode = False
        #print('Reading in {}x{}'.format(width, height))

    def set_original_shape_mode(self, mode):
        self.original_shape_mode = mode

    def reset_init_rescale_factor(self):
        self.set_rescale_factor(self.init_rescale_factor)


class IterableFrameReader:
    def __init__(self, video_filename, skip_frames=0, output_shape=None, progress_bar=False, preload=False, max_frame=0):
        # store arguments for reset
        self.video_filename = video_filename
        self.max_frame_arg = max_frame
        self.progress_bar_arg = progress_bar
        self.preload = preload
        self.skip_frames = skip_frames

        self.video = cv2.VideoCapture(video_filename)
        self.input_shape = (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.total_num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

        self.max_num_frames = min(max_frame, self.total_num_frames) if max_frame!=0 else self.total_num_frames
        self.counter = 0
        self.progress_bar = None

        if output_shape is None:
            w, h = self.input_shape
            new_h = (h | 31) + 1
            new_w = (w | 31) + 1
            self.output_shape = (new_w, new_h)
        else:
            self.output_shape = output_shape

        self.fps = self.video.get(cv2.CAP_PROP_FPS) / (self.skip_frames+1)

        if self.preload:
            self.frames = self._load_all_frames()

    def update_progress_bar(self):
        if self.progress_bar_arg:
            if self.progress_bar:
                # update_progress_bar
                self.progress_bar.update()
            else:
                # create progress bar
                self.progress_bar = tqdm(total=int(self.max_num_frames/(self.skip_frames+1)),
                                         position=1, leave=True)

    def reset_video(self):
        """ This method is needed as cv2.CAP_PROP_POS_FRAMES
        does not work on all backends
        """
        self.video.release()
        self.progress_bar.close()
        self.__init__(self.video_filename, self.skip_frames, self.output_shape,
                      self.progress_bar_arg, self.preload, self.max_frame_arg)

    def _load_all_frames(self):
        frames = []
        while True:
            ret, frame = self._read_frame()
            if ret:
                frames.append(frame)
            else: break
        if self.progress_bar: self.progress_bar.reset()
        return frames

    def __next__(self):
        self.counter+=1
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

        self.reset_video()
        raise StopIteration

    def _read_frame(self):
        ret, frame = self.video.read()
        self._skip_frames()
        if ret:
            self.update_progress_bar()
            frame =  cv2.resize(frame, self.output_shape)
        return ret, frame

    def __iter__(self):
        return self

    def _skip_frames(self):
        for _ in range(self.skip_frames):
            self.counter+=1
            self.video.read()


class SimpleVideoReader:
    def __init__(self, video_filename, skip_frames=0):
        self.skip_frames = skip_frames
        self.video = cv2.VideoCapture(video_filename)
        self.shape = (int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = self.video.get(cv2.CAP_PROP_FPS) / (skip_frames+1)
        self.frame_nb = 0
        self.num_frames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)

    def read(self):
        ret, frame = self.video.read()
        self.frame_nb+=1
        self._skip_frames()
        return ret, frame, self.frame_nb-1

    def set_frame(self, frame_nb_to_set):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_nb_to_set)
        self.frame_nb = frame_nb_to_set

    def _skip_frames(self):
        for _ in range(self.skip_frames):
            self.video.read()


class TorchIterableFromReader(torch.utils.data.IterableDataset):
    def __init__(self, reader, transforms):
        self.transforms = transforms
        self.reader = reader

    def __iter__(self):
        for frame in self.reader:
            yield self.transforms(frame)
