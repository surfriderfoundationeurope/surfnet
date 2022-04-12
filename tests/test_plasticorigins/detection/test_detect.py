from serving.inference import model, device
from plasticorigins.detection.detect import detect

import torch
import numpy as np

preprocessed_frames = torch.load('tests/ressources/pf.pt')
def test_detect():
    res = detect(preprocessed_frames=preprocessed_frames.to(device),
                 threshold=0.3,
                 model=model)
    
    assert len(res) == 1
    assert res[0].shape == (2,2)
    np.testing.assert_array_equal(res[0], np.array([[ 34,  83],[175,  85]]))