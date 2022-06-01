from matplotlib.font_manager import json_load
from src.surfnet_train.data.data_processing import coco2yolo
from src.surfnet_train.data.data_processing import image_orientation
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont, ExifTags
from pycocotools.coco import COCO
import os
import json




def test_coco2yolo():

    bbox = np.array([300, 300, 100, 100])
    output = coco2yolo(bbox, 1000, 1000)

    assert len(output) == 4
    np.testing.assert_array_equal(output, [0.35, 0.35, 0.1, 0.1])



def test_image_orientation():

    file = {"images": [{"id": 1, "file_name": "img3423.jpg"}, {"id": 333, "file_name": "img1119.jpg"}],
     "annotations": [{"id": 4, "image_id": 1, "bbox": [1731, 1200, 145, 338], "category_id": 10},
      {"id": 599, "image_id": 333, "bbox": [468, 1908, 108, 130], "category_id": 1}, 
      {"id": 600, "image_id": 333, "bbox": [1273, 1854, 297, 249], "category_id": 1}, 
      {"id": 601, "image_id": 333, "bbox": [487, 1884, 53, 125], "category_id": 1},
       {"id": 602, "image_id": 333, "bbox": [1260, 1867, 204, 142], "category_id": 1}]}

    f = open("tests/test_surfnet_train/utils/data/file.json", "w")
    json.dump(file, f)
    f.close() # create our own json file with 2 images : "file.json"

    my_image = Image.open(os.path.join('tests/test_surfnet_train/utils/images', 
    COCO('tests/test_surfnet_train/utils/data/file.json').loadImgs(333)[0]['file_name']))
    # create the image in the same way as our function 

    output = image_orientation(my_image) 

    assert type(output) == PIL.Image.Image
    assert my_image._getexif()[orientation] == []