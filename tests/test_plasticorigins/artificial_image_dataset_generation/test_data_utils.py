import json
import os
import cv2
import numpy as np
from src.plasticorigins.training.artificial_image_dataset_generation.utils import center_polygon, scale_img, rotate_img, shift_img, get_bbox_from_polygon

current_dir = os.path.dirname(os.path.abspath(__file__))
functions_dir = os.path.join(
    current_dir, "../src/artificial_dataset_generation")
script_path = os.path.join(functions_dir, "utils.py")
dataset_path = os.path.join(current_dir, "ressources/data_gen")
background_dataset_path = os.path.join(
    current_dir, "ressources/background_images")
result_dataset_path = os.path.join(
    current_dir, "ressources/artificial_data")
result_images_path = os.path.join(
    result_dataset_path, "images")
result_labels_path = os.path.join(
    result_dataset_path, "labels")

def get_img_ann():
    # Check if new x,y == center
    file_name = "000004.jpg"
    img_path = os.path.join(dataset_path, file_name)
    ann_path = os.path.join(dataset_path, "annotations.json")

    image = cv2.imread(img_path)
    H, W = image.shape[:2]
    img_size = (W, H)

    with open(ann_path, 'r') as f:
        dataset = json.loads(f.read())

    img_id = [i['id'] for i in dataset['images']
              if i['file_name'].lower() == file_name]
    # print(img_id)
    img_id = img_id[0]
    annotations = [i for i in dataset['annotations']
                   if i['image_id'] == img_id]

    return image, img_size, annotations


def test_center_polygon():
    image, img_size, annotations = get_img_ann()
    W, H = img_size
    for ann in annotations:
        print(ann['id'])
        seg = ann['segmentation']
        polygon = np.array(np.array(seg), np.int32)
        polygon = polygon.reshape((-1, 2))
        image, polygon = center_polygon(image, polygon)
        bounding_box = get_bbox_from_polygon(polygon, img_size)
        x, y, width, height = bounding_box

        assert (x == W//2)
        assert (y == H//2)


def test_scale():
    image, img_size, annotations = get_img_ann()
    W, H = img_size
    for ann in annotations:
        seg = ann['segmentation']
        polygon = np.array(np.array(seg), np.int32)
        polygon = polygon.reshape((-1, 2))
        image, polygon = scale_img(image, polygon)
        bounding_box = get_bbox_from_polygon(polygon, img_size)
        x, y, width, height = bounding_box

        assert (width < 0.15*W) or (height < 0.15*H)
        assert (width > 0.05*W) or (height > 0.05*H)


def test_shift():
    image, img_size, annotations = get_img_ann()
    W, H = img_size
    for ann in annotations:
        seg = ann['segmentation']
        polygon = np.array(np.array(seg), np.int32)
        polygon = polygon.reshape((-1, 2))
        image, polygon = shift_img(image, polygon)
        bounding_box = get_bbox_from_polygon(polygon, img_size)
        x, y, width, height = bounding_box

        assert (x >= width//2)
        assert (x <= W - width//2)
        assert (y >= height//2)
        assert (y <= H - height//2)
