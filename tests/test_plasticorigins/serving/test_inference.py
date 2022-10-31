import json
import math
import os
from pathlib import Path

import pytest
from werkzeug.datastructures import FileStorage

from plasticorigins.tools.files import create_unique_folder
from plasticorigins.serving.app import app
from plasticorigins.serving.config import config_track
from plasticorigins.serving.inference import track

video_file = "tests/ressources/validation_videos/T1_trim.mp4"


@pytest.fixture
def client():
    return app.test_client()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_inference(client):

    video = video_file
    file = FileStorage(
        stream=open(video, "rb"),
        filename="T1_trim.mp4",
        content_type="video/mpeg",
    )

    data = {"file": file}
    resp = client.post("/", content_type="multipart/form-data", data=data)

    resp_data = json.loads(resp.get_data())

    assert resp.status_code == 200
    assert list(resp_data.keys()) == [
        "detected_trash",
        "fps",
        "video_id",
        "video_length",
    ]
    if len(resp_data["detected_trash"]) > 0:
        assert set(resp_data["detected_trash"][0].keys()) == {
            "frame_to_box",
            "id",
            "label",
            "avg_conf",
        }
        assert type(resp_data["detected_trash"][0]["id"]) == int
        assert type(resp_data["detected_trash"][0]["label"]) == str
        assert type(resp_data["detected_trash"][0]["frame_to_box"]) == dict


def test_inference_no_file(client):
    data = {}
    try:
        client.post("/", content_type="multipart/form-data", data=data)
    except FileNotFoundError:
        assert True


def test_track():

    filename = video_file.split("/")[-1]
    video = video_file
    file = FileStorage(
        stream=open(video, "rb"),
        filename="T1_trim.mp4",
        content_type="video/mpeg",
    )

    working_dir = Path(create_unique_folder(config_track.upload_folder, filename))
    full_filepath = working_dir / filename
    if os.path.isfile(full_filepath):
        os.remove(full_filepath)
    file.save(full_filepath)
    config_track.video_path = full_filepath.as_posix()
    config_track.output_dir = working_dir.as_posix()

    # launch the tracking
    filtered_results, num_frames, fps = track(config_track)
    os.remove(full_filepath)
    assert math.isclose(fps, 5.921, abs_tol=0.01)
    assert num_frames == 34
    assert len(filtered_results) == 7
    assert len(filtered_results[0]) == 6
    assert type(filtered_results[0][0]) == int
    assert type(filtered_results[0][1]) == int
    # assert type(filtered_results[0][2]) == np.float64
    # assert type(filtered_results[0][3]) == np.float64
    # assert type(filtered_results[0][4]) == np.float64
    assert type(filtered_results[0][5]) == int
