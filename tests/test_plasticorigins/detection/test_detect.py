import numpy as np
import torch

from plasticorigins.detection.detect import detect
from plasticorigins.detection.centernet.models import load_model_simple

preprocessed_frames = torch.load("tests/ressources/pf.pt")


def test_detect():
    model = load_model_simple()

    res = detect(
        preprocessed_frames=preprocessed_frames,
        threshold=0.3,
        model=model,
    )

    assert len(res) == 1
    assert res[0].shape == (3, 2)
    np.testing.assert_array_equal(
        res[0], np.array([[149, 76], [33, 83], [177, 85]])
    )
