from plasticorigins.tracking.utils import read_tracking_results
from plasticorigins.tracking.count import (
    count_detected_objects,
    video_count_truth,
)

video_count_path = "tests/ressources/validation_videos/T1_trim.txt"
tracking_results_file = "tests/ressources/T1_trim_filtered_results.txt"


def test_evaluate_detected_count():
    results = read_tracking_results(tracking_results_file)
    n_det, _ = count_detected_objects(results, video_count_path, compare=False)
    assert n_det == 1


def test_video_count_truth():
    n = video_count_truth(video_count_path)
    assert n == 2
