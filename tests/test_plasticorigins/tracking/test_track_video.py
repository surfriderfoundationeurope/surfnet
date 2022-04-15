import os

import numpy as np

from plasticorigins.tools.video_readers import IterableFrameReader
from plasticorigins.tracking.track_video import track_video
from serving.inference import config_track, engine


def test_track_video():
    config_track.video_path = "tests/ressources/validation_videos/T1_trim.mp4"
    detections = np.load("tests/ressources/detections.npy", allow_pickle=True)
    detections = list(detections)

    transition_variance = np.load(
        os.path.join(
            config_track.noise_covariances_path, "transition_variance.npy"
        )
    )
    observation_variance = np.load(
        os.path.join(
            config_track.noise_covariances_path, "observation_variance.npy"
        )
    )

    reader = IterableFrameReader(
        video_filename=config_track.video_path,
        skip_frames=config_track.skip_frames,
        output_shape=config_track.output_shape,
        progress_bar=True,
        preload=config_track.preload_frames,
    )

    results = track_video(
        reader,
        iter(detections),
        config_track,
        engine,
        transition_variance,
        observation_variance,
        None,
    )
    assert len(results) == 16
    assert len(results[0]) == 6
