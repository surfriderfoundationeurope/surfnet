import numpy as np
import torch

from plasticorigins.detection.detect import detect
from plasticorigins.tools.misc import load_model

preprocessed_frames = torch.load("tests/ressources/pf.pt")


def test_detect():
    model = load_model(
        arch="mobilenet_v3_small", model_weights="models/mobilenet_v3_pretrained.pth", device=device,
    )

    res = detect(
        preprocessed_frames=preprocessed_frames.to(device),
        threshold=0.3,
        model=model,
    )

    assert len(res) == 1
    assert res[0].shape == (2, 2)
    np.testing.assert_array_equal(res[0], np.array([[34, 83], [175, 85]]))
