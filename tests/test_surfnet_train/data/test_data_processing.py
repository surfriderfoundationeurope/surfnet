from surfnet_train.data.data_processing import coco2yolo
import numpy as np

def test_coco2yolo():

    bbox = np.array([300, 300, 100, 100])
    output = coco2yolo(bbox, 1000, 1000)

    assert len(output) == 4
    np.testing.assert_array_equal(output, [0.35, 0.35, 0.1, 0.1])









