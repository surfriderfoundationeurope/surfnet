import numpy as np

from plasticorigins.detection.centernet.networks.mobilenet import MobiletNetHM
from plasticorigins.tools.misc import (blob_for_bbox, load_model)
from plasticorigins.serving.config import config_track


def test_load_model():
    model_mbn = load_model(
        arch="mobilenet_v3_small", model_weights=None, device="cpu"
    )
    assert type(model_mbn) == MobiletNetHM


def test_blob_for_bbox():
    heatmap = np.random.rand(500, 500)
    bbox = [100, 100, 20, 20]
    heatmap_res, cint = blob_for_bbox(
        heatmap=heatmap.copy(), downsampling_factor=None, bbox=bbox
    )
    assert not np.array_equal(
        heatmap_res[100:120, 100:120], heatmap[100:120, 100:120]
    )
    assert cint[0] == 110 and cint[1] == 110
