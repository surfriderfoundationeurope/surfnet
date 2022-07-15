import numpy as np
import torch

from plasticorigins.tools.files import download_from_url
from plasticorigins.detection.yolo import load_model, predict_yolo
from argparse import Namespace
import cv2

frame = cv2.imread("tests/ressources/test_image.jpeg")
config_track = Namespace(
    yolo_conf_thrld= 0.35,
    yolo_iou_thrld= 0.5,
    url_model_yolo= "https://github.com/surfriderfoundationeurope/IA_Pau/releases/download/v0.1/yolov5.pt",
    file_model_yolo= "yolov5.pt",
    output_shape= (960, 544),
    size= 768,
    device= "cpu",
)

def test_detect():
    model_path = download_from_url(config_track.url_model_yolo, config_track.file_model_yolo, "./models", logger)
    model_yolo = load_model(model_path, config_track.device, config_track.yolo_conf_thrld, config_track.yolo_iou_thrld)
    res = predict_yolo(model_yolo, frame, config_track.size, augment=False)

    assert len(res) == 1
    assert res[0].shape == (2, 2)
    np.testing.assert_array_equal(res[0], np.array([[34, 83], [175, 85]]))
