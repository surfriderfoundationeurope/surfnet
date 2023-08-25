import cv2
import os
import numpy as np
import json
import seaborn as sns
from PIL import ExifTags
from pycocotools.coco import COCO
import pylab
import shutil
import argparse
from categories_map import categories_map
from utils import get_background_names, transform_img, get_bbox_from_contour, paste_shape, generate_image_identifier
import concurrent.futures


def create_img(image_path, seg):
    """
    Create a new image by compositing a shape onto a target image.

    Args:
        image_path (str): Path to the input image.
        seg (list): List of polygon coordinates.

    Returns:
        tuple: A tuple containing the shape to be pasted and its bounding box.
    """

    try:
        # Load image and annotation file
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        img_size = (w, h)

        # Extract polygon coordinates from annotation data
        polygon = np.array(np.array(seg), np.int32)
        polygon = polygon.reshape((-1, 2))
        image, polygon = transform_img(image, polygon)

        try:
            # Create a binary mask from the polygon
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            shape_mask = np.zeros_like(mask)
        except cv2.error as e:
            print("OpenCV error occurred:", str(e))
        finally:
            # Find the contour
            contour = polygon.reshape((-1, 1, 2)).astype(np.int32)
            box = get_bbox_from_contour(contour, img_size)

            # Paste the contour on the mask
            cv2.drawContours(shape_mask, [contour], 0, 255, -1)
            # Apply the mask to the original image to extract the shape
            shape = cv2.bitwise_and(image, image, mask=shape_mask)
            return shape, box

    except Exception as e:
        print(str(e))


def create_labels(label_path, anns, img_size):
    """
    Create a label file for an image.

    Args:
        label_path (str): Path to the label file.
        anns (list): list of object annotations.
        img_size (tuple): Image size (width, height).
    """

    with open(label_path, "w") as file:
        for ann in anns:
            bbox = ann['bbox']
            category_id = ann['category_id']

            # transform the bbox values from number of pixels to percentage of image width and height
            w, h = img_size
            x, y, width, height = bbox
            bbox = [x / w, y / h, width / w, height / h]

            # write the lbl line to the file
            # the label line contains the category_id and the values of the bbox
            lbl = f"{category_id} {' '.join(str(x) for x in bbox)}\n"
            file.write(lbl)


def create_new_img(image_path, target_img_path, res_dataset_path, result_labels_path, input_anns, categories_dict):
    """
    Create a new image and its corresponding label.

    Args:
        image_path (str): Path to the input image.
        target_img_path (str): Path to the target image.
        res_dataset_path (str): Path to save the resulting images.
        result_labels_path (str): Path to save the resulting labels.
        input_anns (list): List of object annotations.
        categories_dict (dict): Dictionnary that maps Plastic Origins and TACO dataset classes.
    """
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    img_size = (w, h)
    # Load and resize the target image
    target = cv2.imread(target_img_path)
    target = cv2.resize(target, img_size)

    # Create a blank canvas & Paste the target image onto the canvas
    canvas = np.zeros_like(target)
    canvas[0:target.shape[0], 0:target.shape[1]] = target

    # print(anns)
    anns = []
    for ann in input_anns:
        # print(ann)
        try:
            seg = ann['segmentation']
            category_id = categories_dict[ann['category_id']]

            # create the new image, contour, area & bbox
            shape, bbox = create_img(image_path, seg)
            # Paste the shape onto the canvas on top of the target image
            canvas = paste_shape(shape, canvas)
            # TODO
            bbox = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]
            anns.append({'category_id': category_id, 'bbox': bbox})
            # print(str(ann['id']) + " done")

        except Exception as e:
            print(
                f"An error occurred with {ann['id']} in the image '{image_path}': {e}")
            continue

    # Save the result as a new image
    img_id = generate_image_identifier(canvas, image_path)
    filename = img_id + '.jpg'
    res_path = res_dataset_path + '/' + filename
    cv2.imwrite(res_path, canvas)
    print("drawing image...")

    # create the image annotations
    label_file_name = img_id + '.txt'
    label_path = result_labels_path + '/' + label_file_name
    create_labels(label_path, anns, img_size)
    print("annotations ...")


def main(dataset_path,
         background_dataset_path,
         result_dataset_path):
    """
    Main function to create artificial images and labels.
    Args:
        dataset_path (str): Path to the dataset.
        background_dataset_path (str): Path to the background dataset.
        result_dataset_path (str): Path to save the resulting dataset.
    """
    sns.set()

    # Paths
    anns_file_path = dataset_path + '/annotations.json'
    result_images_path = result_dataset_path + '/images'
    result_labels_path = result_dataset_path + "/labels"

    if shutil.os.path.exists(result_dataset_path):
        shutil.rmtree(result_dataset_path)

    # create the save directories if they do not exist
    if not os.path.exists(result_dataset_path):
        os.makedirs(result_dataset_path)
    if not os.path.exists(result_images_path):
        os.makedirs(result_images_path)
    if not os.path.exists(result_labels_path):
        os.makedirs(result_labels_path)

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    # Getting information from the json file
    imgs = dataset['images']

    categories_dict = {}
    totalCatIds = []
    for category_name in categories_map.keys():
        pylab.rcParams['figure.figsize'] = (14, 14)

        # Obtain Exif orientation tag code
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        # Loads dataset as a coco object
        coco = COCO(anns_file_path)

        # Get image ids
        catIds = coco.getCatIds(catNms=[category_name])
        for c in catIds:
            categories_dict[c] = categories_map[category_name][0]
        totalCatIds += catIds
        if not catIds:
            # Get all images containing an instance of the chosen super category
            catIds = coco.getCatIds(supNms=[category_name])

            for c in catIds:
                categories_dict[c] = categories_map[category_name][0]
            totalCatIds += catIds

    # print(categories_dict)
    # Get all images containing an instance of the chosen category
    imgIds = []
    # print(totalCatIds)
    for catId in totalCatIds:
        imgIds += (coco.getImgIds(catIds=catId))

    imgIds = list(set(imgIds))
    imgs = coco.loadImgs(imgIds)
    # print(imgIds)

    tasks = []
    for img in imgs:
        image_path = dataset_path + '/' + img['file_name']
        target_images = get_background_names(background_dataset_path)
        for target_img in target_images:
            target_img_path = background_dataset_path + '/' + target_img

            # Load mask ids
            annIds = coco.getAnnIds(
                imgIds=img['id'], catIds=totalCatIds, iscrowd=None)
            anns_sel = coco.loadAnns(annIds)
            # create_new_img(image_path, target_img_path, result_images_path, result_labels_path, anns_sel, categories_dict)
            tasks.append((image_path, target_img_path,
                          result_images_path, result_labels_path, anns_sel, categories_dict))

        # Using multithreading to parallelize create_new_img calls

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(create_new_img_wrapper, tasks)


def create_new_img_wrapper(args):
    """
    Wrapper function for creating new images with parallel processing.

    Args:
        args (tuple): Tuple containing the input arguments for creating a new image.
    """
    image_path, target_img_path, result_images_path, result_labels_path, anns_sel, categories_dict = args
    create_new_img(image_path, target_img_path,
                   result_images_path, result_labels_path, anns_sel, categories_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define default paths
    default_dataset_path = '../data_TACO/data'
    default_background_dataset_path = '../extracted_background_images'
    default_result_dataset_path = '../artificial_data_n_objects'

    # Add arguments
    parser.add_argument("--dataset_path", type=str,
                        default=default_dataset_path, help="Path to the dataset")
    parser.add_argument("--background_dataset_path", type=str,
                        default=default_background_dataset_path, help="Path to the background dataset")
    parser.add_argument("--result_dataset_path", type=str,
                        default=default_result_dataset_path, help="Path to save the resulting dataset")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    background_dataset_path = args.background_dataset_path
    result_dataset_path = args.result_dataset_path

    # Call the main function and pass the parameters
    main(dataset_path, background_dataset_path, result_dataset_path)
