import numpy as np
from PIL import Image

from plasticorigins.detection.transforms import TrainTransforms, ValTransforms

image = np.array(Image.open("tests/ressources/test_image.jpeg"))
target = {"bboxes": [[100, 100, 20, 20]], "cats": [1]}


def test_train_transforms():
    trtf = TrainTransforms(540, (544, 960), 1, 4)
    img, target_ = trtf(image, target)
    assert np.array_equal(img.shape, [3, 544, 960])


def test_val_transforms():
    valtf = ValTransforms(540, (544, 960), 1, 4)
    img, target_ = valtf(image, target)
    assert np.array_equal(img.shape, [3, 544, 960])
