import argparse
import cv2
import os
import numpy as np
import json
import seaborn as sns
from PIL import ExifTags
from pycocotools.coco import COCO
import pylab
import sys
import argparse
import shutil
import random
from categories_map import categories_map
from utils import get_background_names, transform_img, get_bbox_from_contour, paste_shape, generate_image_identifier
import concurrent.futures


def pick_items(lst, i, max_picks_per_item):
    n = len(lst)
    i = min(max_picks_per_item * n, i)
    k = i//n

    # Calculate the number of items to pick from each element
    min_picks = [k] * n
    remaining_picks = i - (k * n)
    for j in range(remaining_picks):
        min_picks[j] += 1

    # Shuffle the list and select items based on the constraints
    random.shuffle(lst)
    result = []
    for item, min_pick in zip(lst, min_picks):
        result.extend(random.sample([item]*min_pick, min_pick))

    return result


def pick_random_element(lst):
    if not lst:
        return None

    random_element = random.choice(lst)
    return random_element


def extract_shape(image, polygon):
    h, w = image.shape[:2]
    img_size = (w, h)
    try:
        # Create a binary mask from the polygon
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        shape_mask = np.zeros_like(mask)
    except cv2.error as e:
        print("OpenCV error occurred:", str(e))
    except:
        # Catch any uncaught exception and print its type
        print("An uncaught exception occurred:")
        exc_type, _, _ = sys.exc_info()
        print("Exception type:", exc_type)
    finally:
        # Find the contour
        contour = polygon.reshape((-1, 1, 2)).astype(np.int32)
        box = get_bbox_from_contour(contour, img_size)

        shape_mask = np.zeros_like(mask)
        cv2.drawContours(shape_mask, [contour], 0, 255, -1)

        # Apply the mask to the original image to extract the shape
        shape = cv2.bitwise_and(image, image, mask=shape_mask)
    return shape, box


def create_img(image_path, seg, result_images_path, target_img_path):
    # Load image and annotation file
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    img_size = (w, h)

    # Extract polygon coordinates from annotation data
    polygon = np.array(np.array(seg), np.int32)
    polygon = polygon.reshape((-1, 2))
    image, polygon = transform_img(image, polygon)
    shape, box = extract_shape(image, polygon)

    # Load and resize the target image
    target = cv2.imread(target_img_path)
    target = cv2.resize(target, img_size)

    # Create a blank canvas of the same size as the target image
    canvas = np.zeros_like(target)
    # Paste the target image onto the canvas
    canvas[0:target.shape[0], 0:target.shape[1]] = target

    # Paste the shape onto the canvas on top of the target image
    canvas = paste_shape(shape, canvas)

    # Save the result as a new image
    img_id = generate_image_identifier(canvas, image_path)
    file_name = img_id + '.jpg'
    res_path = result_images_path + '/' + file_name
    cv2.imwrite(res_path, canvas)
    return img_id, box, img_size


def create_label(label_path, bbox, category_id, img_size):
    # transform the bbox values from number of pixels to percentage of image width and height
    w, h = img_size
    x, y, width, height = bbox
    bbox = [x/w, y/h, width/w, height/h]

    with open(label_path, "w") as file:
        lbl = f"{category_id} {' '.join(str(x) for x in bbox)}\n"
        file.write(lbl)


def create_new_img(image_path, target_img_path, result_images_path, result_labels_path, seg, category_id):

    # create the new image & get the bbox
    img_id, bbox, img_size = create_img(
        image_path, seg, result_images_path, target_img_path)

    label_file_name = img_id + '.txt'
    label_path = result_labels_path + '/' + label_file_name
    create_label(label_path, bbox, category_id, img_size)


def create_new_img_wrapper(args):
    image_path, target_img_path, result_images_path, result_labels_path, seg, cat_id = args
    create_new_img(image_path, target_img_path,
                   result_images_path, result_labels_path, seg, cat_id)


def main(dataset_path,
         background_dataset_path,
         result_dataset_path, num_uses_background):
    sns.set()

    num_uses_background = int(num_uses_background)
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

    for category_name in categories_map.keys():
        category_id = categories_map[category_name][0]
        pylab.rcParams['figure.figsize'] = (14, 14)

        # Obtain Exif orientation tag code
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        # Loads dataset as a coco object
        coco = COCO(anns_file_path)

        # Get image ids
        imgIds = []
        catIds = coco.getCatIds(catNms=[category_name])
        if catIds:
            # Get all images containing an instance of the chosen category
            imgIds = coco.getImgIds(catIds=catIds)
        else:
            # Get all images containing an instance of the chosen super category
            catIds = coco.getCatIds(supNms=[category_name])
            for catId in catIds:
                imgIds += (coco.getImgIds(catIds=catId))
            imgIds = list(set(imgIds))

        imgs = coco.loadImgs(imgIds)

        tasks = []
        for img in imgs:
            image_path = dataset_path + '/' + img['file_name']
            target_images = get_background_names(background_dataset_path)

            target_img = pick_random_element(target_images)
            picked_target_imgs = pick_items(
                target_images, num_uses_background, 1)
            for target_img in picked_target_imgs:
                target_img_path = background_dataset_path + '/' + target_img
                print(image_path, target_img_path)
                # Load mask ids
                annIds = coco.getAnnIds(
                    imgIds=img['id'], catIds=catIds, iscrowd=None)
                anns_sel = coco.loadAnns(annIds)

                # Show annotations
                for ann in anns_sel:
                    for seg in ann['segmentation']:
                        # create_new_img(  image_path, target_img_path, result_images_path, result_labels_path, seg, category_id)
                        tasks.append((image_path, target_img_path,
                                      result_images_path, result_labels_path, seg, category_id))

         # Using multithreading to parallelize create_new_img calls
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(create_new_img_wrapper, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define default paths
    default_dataset_path = '../data_TACO/data'
    default_background_dataset_path = '../extracted_background_images'
    default_result_dataset_path = '../artificial_data'
    default_num_uses_background = 6

    # Add arguments
    parser.add_argument("--dataset_path", type=str,
                        default=default_dataset_path, help="Path to the dataset")
    parser.add_argument("--background_dataset_path", type=str,
                        default=default_background_dataset_path, help="Path to the background dataset")
    parser.add_argument("--result_dataset_path", type=str,
                        default=default_result_dataset_path, help="Path to save the resulting dataset")
    parser.add_argument("--num_uses_background", type=str,
                        default=default_num_uses_background, help="The number of background images that are used with each object")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    background_dataset_path = args.background_dataset_path
    result_dataset_path = args.result_dataset_path
    num_uses_background = args.num_uses_background

    # Call the main function and pass the parameters
    main(dataset_path, background_dataset_path,
         result_dataset_path, num_uses_background)
