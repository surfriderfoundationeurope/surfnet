import numpy as np

from plasticorigins.tracking.postprocess_and_count_tracks import (
    filter_tracks,
    postprocess_for_api,
)
from plasticorigins.serving.config import id_categories

results = np.load("tests/ressources/results_to_filter.npy", allow_pickle=True)


def test_filter_tracks():
    filtered_results = filter_tracks(results, kappa=5, tau=3)
    assert len(filtered_results) == 53
    assert len(filtered_results[0]) == 6
    assert type(filtered_results[0][0]) == int
    assert type(filtered_results[0][1]) == int
    assert (
        type(filtered_results[0][2]) == float
        or type(filtered_results[0][2]) == np.float64
    )
    assert (
        type(filtered_results[0][3]) == float
        or type(filtered_results[0][2]) == np.float64
    )
    assert (
        type(filtered_results[0][4]) == float
        or type(filtered_results[0][2]) == np.float64
    )


def test_post_process_for_api():
    filtered_results = filter_tracks(results, kappa=5, tau=3)
    resp_data = postprocess_for_api(filtered_results, id_categories)
    assert list(resp_data.keys()) == ["detected_trash"]
    if len(resp_data["detected_trash"]) > 0:
        assert set(resp_data["detected_trash"][0].keys()) == {
            "label",
            "id",
            "frame_to_box",
            "avg_conf",
        }
        assert type(resp_data["detected_trash"][0]["id"]) == int
        assert type(resp_data["detected_trash"][0]["label"]) == str
        assert type(resp_data["detected_trash"][0]["frame_to_box"]) == dict
