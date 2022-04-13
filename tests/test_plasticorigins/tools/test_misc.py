from plasticorigins.tools.misc import load_model, blob_for_bbox, pre_process_centernet
from plasticorigins.detection.centernet.networks.mobilenet import MobiletNetHM
from serving.config import config_track
import numpy as np
from PIL import Image
from torch import Tensor


def test_load_model():
    model_mbn = load_model(arch=config_track.arch, model_weights=None, device="cpu")
    assert type(model_mbn) == MobiletNetHM


def test_blob_for_bbox():
    heatmap = np.random.rand(500, 500)
    bbox = [100, 100, 20, 20]
    heatmap_res, cint = blob_for_bbox(
        heatmap=heatmap.copy(), downsampling_factor=None, bbox=bbox
    )
    assert not np.array_equal(heatmap_res[100:120, 100:120], heatmap[100:120, 100:120])
    assert cint[0] == 110 and cint[1] == 110


def test_pre_process_centernet():
    image = np.array(Image.open("tests/ressources/test_image.jpeg"))
    image_res = pre_process_centernet(image=image)
    assert np.array_equal(image_res.shape, [3, 512, 512])
    assert type(image_res) == Tensor
