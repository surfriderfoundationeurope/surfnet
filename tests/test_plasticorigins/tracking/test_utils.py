import os

import numpy as np
from scipy.stats._multivariate import multivariate_normal_frozen

from plasticorigins.detection.detect import detect
from plasticorigins.tools.video_readers import IterableFrameReader
from plasticorigins.tracking.utils import (
    GaussianMixture,
    exp_and_normalize,
    gather_filenames_for_video_in_annotations,
    get_detections_for_video,
    overlay_transparent,
    read_tracking_results,
    write_tracking_results_to_file,
)
from plasticorigins.serving.inference import config_track, device, model

results = np.load("tests/ressources/results.npy", allow_pickle=True)
results = [tuple(res) for res in results]


def test_get_detections_for_video():
    config_track.video_path = "tests/ressources/validation_videos/T1_trim.mp4"
    detector = lambda frame: detect(frame, threshold=0.3, model=model)
    reader = IterableFrameReader(
        video_filename=config_track.video_path,
        skip_frames=config_track.skip_frames,
        output_shape=config_track.output_shape,
        progress_bar=True,
        preload=config_track.preload_frames,
    )

    detections = get_detections_for_video(
        reader,
        detector,
        batch_size=config_track.detection_batch_size,
        device=device,
    )

    assert len(detections) == 9


def test_write_tracking_results_to_file():
    input_shape = (640, 360)
    output_shape = (960, 544)
    ratio_y = input_shape[0] / (
        output_shape[0] // config_track.downsampling_factor
    )
    ratio_x = input_shape[1] / (
        output_shape[1] // config_track.downsampling_factor
    )

    tmp_folder = "tests/ressources/tmp"
    output_filename = os.path.join(tmp_folder, "results.txt")
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    write_tracking_results_to_file(
        results,
        ratio_x=ratio_x,
        ratio_y=ratio_y,
        output_filename=output_filename,
    )

    assert os.path.exists(output_filename)

    os.remove(output_filename)
    os.rmdir(tmp_folder)


def test_read_tracking_results():
    input_shape = (640, 360)
    output_shape = (960, 544)
    ratio_y = input_shape[0] / (
        output_shape[0] // config_track.downsampling_factor
    )
    ratio_x = input_shape[1] / (
        output_shape[1] // config_track.downsampling_factor
    )

    tmp_folder = "tests/ressources/tmp"
    output_filename = os.path.join(tmp_folder, "results.txt")
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)

    write_tracking_results_to_file(
        results,
        ratio_x=ratio_x,
        ratio_y=ratio_y,
        output_filename=output_filename,
    )

    results_read = read_tracking_results(output_filename)
    os.remove(output_filename)

    os.rmdir(tmp_folder)

    assert 1 + int(max(r[1] for r in results)) == len(results_read)


def test_gaussian_mixture():
    means = np.random.randn(5)
    weights = np.random.randn(5)
    cov = 0.5
    gmm = GaussianMixture(means=means, weights=weights, covariance=cov)

    x = np.random.randn(5)
    results_pdf = gmm.pdf(x)
    results_cdf = gmm.cdf(x)
    results_lpdf = gmm.logpdf(x)

    assert len(gmm.components) == len(means)
    assert type(gmm.components[0]) == multivariate_normal_frozen
    assert np.array_equal(gmm.weights, weights)

    assert len(results_pdf) == len(x)
    assert len(results_cdf) == len(x)
    assert np.array_equal(results_lpdf, np.log(results_pdf), equal_nan=True)


def test_exp_and_normalize():
    x = 2 + 0.5 * np.random.randn(5)
    res = exp_and_normalize(x)
    expected_res = np.exp(x - x.max())
    expected_res = expected_res / expected_res.sum()
    assert np.array_equal(res, expected_res)


def test_gather_filename_for_video_annotations():

    images = [
        {"frame_id": i, "video_id": 1, "file_name": f"1_{i}_.mp4"}
        for i in range(10)
    ]
    images.extend(
        [
            {"frame_id": i, "video_id": 2, "file_name": f"2_{i}_.mp4"}
            for i in range(10)
        ]
    )
    video = {
        "id": 1,
    }
    data_dir = "ressources"

    file_paths = gather_filenames_for_video_in_annotations(
        video, images, data_dir
    )
    dir_ = np.unique([p.split("/")[0] for p in file_paths])[0]
    video_id = np.unique([p.split("/")[1].split("_")[0] for p in file_paths])[
        0
    ]
    assert len(file_paths) == 10
    assert dir_ == data_dir
    assert int(video_id) == video.get("id")


def test_overlay_transparent():
    background = np.random.rand(500, 500, 3)
    overlay = np.zeros((100, 100, 3))
    overlay_1 = np.ones((100, 100, 4))
    overlay_1[:, :, 3] = 255 * np.ones((100, 100))
    background_1 = overlay_transparent(
        background=background.copy(), overlay=overlay, x=100, y=100
    )

    background_2 = overlay_transparent(
        background=background.copy(), overlay=overlay, x=600, y=100
    )

    background_3 = overlay_transparent(
        background=background.copy(), overlay=overlay, x=450, y=450
    )

    background_4 = overlay_transparent(
        background=background.copy(), overlay=overlay_1, x=100, y=100
    )

    assert np.array_equal(background_1[100:200, 100:200, :], overlay)
    assert np.array_equal(background_2, background)
    assert np.array_equal(background_3[450:, 450:, :], overlay[:50, :50, :])
    assert np.array_equal(background_4[100:200, 100:200], overlay_1[:, :, :3])
