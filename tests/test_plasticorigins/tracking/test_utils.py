from serving.inference import model, config_track, device
from plasticorigins.detection.detect import detect
from plasticorigins.tools.video_readers import IterableFrameReader
from plasticorigins.tracking.utils import get_detections_for_video, write_tracking_results_to_file

import numpy as np
import os


results = np.load('tests/ressources/results.npy',allow_pickle=True)
results = [tuple(res) for res in results]

def test_get_detections_for_video():
    config_track.video_path = 'tests/ressources/validation_videos/T1_720_px_converted_trim.mp4'
    detector = lambda frame: detect(frame, threshold=0.3, model=model)
    reader = IterableFrameReader(video_filename=config_track.video_path,
                                    skip_frames=config_track.skip_frames,
                                    output_shape=config_track.output_shape,
                                    progress_bar=True,
                                    preload=config_track.preload_frames)


    detections = get_detections_for_video(reader, detector, batch_size=config_track.detection_batch_size, device=device)

    assert len(detections) == 9


def test_write_tracking_results_to_file():
    input_shape = (640, 360)
    output_shape = (960, 544)
    ratio_y = input_shape[0] / (output_shape[0] // config_track.downsampling_factor)
    ratio_x = input_shape[1] / (output_shape[1] // config_track.downsampling_factor)

    tmp_folder = 'tests/ressources/tmp'
    output_filename = os.path.join(tmp_folder,'results.txt')
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)



    write_tracking_results_to_file(results, ratio_x=ratio_x, ratio_y=ratio_y, output_filename=output_filename)

    assert os.path.exists(output_filename)


    os.remove(output_filename)
    os.rmdir(tmp_folder)
