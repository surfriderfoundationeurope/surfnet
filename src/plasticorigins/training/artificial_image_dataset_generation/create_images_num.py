import random
import csv
import argparse
import cv2
import os
import numpy as np
import json
import seaborn as sns
from PIL import ExifTags
from pycocotools.coco import COCO
import pylab
import time
import shutil
from categories_map import categories_map
from utils import get_background_names, transform_img, get_bbox_from_contour, paste_shape, generate_image_identifier
import concurrent.futures


def read_csv_to_map(file_path):
    """Reads a CSV file and returns its content as a dictionary.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: A dictionary where keys are from the first column and values are from the second column.
    """

    data_map = {}
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        # header = next(csv_reader)
        # Read the header row
        next(csv_reader)
        for row in csv_reader:
            key = row[0]  # Assuming the first column as the key
            values = row[1]  # Assuming the remaining columns as values

            data_map[key] = int(values)

    return data_map


def pick_items(lst, i, max_picks_per_item):
    """
    Pick items from a list with a maximum limit per item. The function ensures that each item is picked almost the same number of times.

    Args:
        lst (list): The input list from which to pick items.
        i (int): The total number of items to pick.
        max_picks_per_item (int): The maximum number of times an item can be picked.

    Returns:
        list: A list of picked items.
    """

    n = len(lst)
    i = min(max_picks_per_item * n, i)
    k = i // n

    # Calculate the number of items to pick from each element
    min_picks = [k] * n
    remaining_picks = i - (k * n)
    for j in range(remaining_picks):
        min_picks[j] += 1

    # Shuffle the list and select items based on the constraints
    random.shuffle(lst)
    result = []
    for item, min_pick in zip(lst, min_picks):
        result.extend(random.sample([item] * min_pick, min_pick))

    return result


def pick_random_element(lst):
    """
    Pick a random element from a list.

    Args:
        lst (list): The input list.

    Returns:
        Any: A randomly selected element from the list, or None if the list is empty.
    """
    if not lst:
        return None

    random_element = random.choice(lst)
    return random_element


def find_closest_number(input_num):
    """Finds the closest multiple of 10 to the given number.

    Args:
        input_num (int): The input number.

    Returns:
        int: The closest multiple of 10 to the input number.
    """

    if input_num < 10:
        return 10
    elif input_num < 1000:
        magnitude = 10 ** (len(str(input_num)) - 1)
    else:
        magnitude = 10 ** (len(str(input_num)) - 2)
    rounded_num = round(input_num / magnitude) * magnitude
    return rounded_num


def get_needed_label_num(csv_path):

    """Calculates the number of needed labels for each category based on a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing label information.

    Returns:
        dict: A dictionary where keys are category names and values are the number of needed labels.
    """
    labels_map = read_csv_to_map(csv_path)
    desired_num = max(labels_map.values())
    closest_number = find_closest_number(desired_num)

    needed_label_num = {}
    range_labels = 100
    for key in labels_map:
        rand = int(random.uniform(closest_number - range_labels // 2, closest_number + range_labels // 2))
        needed_label_num[key] = max(rand - labels_map[key], 0)
    return needed_label_num


def extract_shape(image, polygon):
    """
    Extract the exact shape of an object from an image using a polygon.

    Args:
        image (numpy.ndarray): The input image.
        polygon (numpy.ndarray): The polygon coordinates.

    Returns:
        tuple: A tuple containing the extracted shape and its bounding box.
    """

    h, w = image.shape[:2]
    img_size = (w, h)
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

        shape_mask = np.zeros_like(mask)
        cv2.drawContours(shape_mask, [contour], 0, 255, -1)

        # Apply the mask to the original image to extract the shape
        shape = cv2.bitwise_and(image, image, mask=shape_mask)
    return shape, box


def create_img(image_path, seg, result_images_path, target_img_path):
    """
    Create a new image by compositing a shape onto a target image.

    Args:
        image_path (str): Path to the input image.
        seg (list): List of polygon coordinates.
        result_images_path (str): Path to save the resulting images.
        target_img_path (str): Path to the target image.

    Returns:
        tuple: A tuple containing the image identifier, bounding box, and image size.
    """

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
    """
    Create a label file for an image.

    Args:
        label_path (str): Path to the label file.
        bbox (list): Bounding box coordinates.
        category_id (int): Category ID.
        img_size (tuple): Image size (width, height).
    """

    # transform the bbox values from number of pixels to percentage of image width and height
    w, h = img_size
    x, y, width, height = bbox
    bbox = [x / w, y / h, width / w, height / h]

    with open(label_path, "w") as file:
        lbl = f"{category_id} {' '.join(str(x) for x in bbox)}\n"
        file.write(lbl)


def create_new_img(image_path, target_img_path, result_images_path, result_labels_path, seg, category_id):
    """
    Create a new image and its corresponding label.

    Args:
        image_path (str): Path to the input image.
        target_img_path (str): Path to the target image.
        result_images_path (str): Path to save the resulting images.
        result_labels_path (str): Path to save the resulting labels.
        seg (list): List of polygon coordinates.
        category_id (int): Category ID.
    """

    # create the new image & get the bbox
    img_id, bbox, img_size = create_img(
        image_path, seg, result_images_path, target_img_path)

    label_file_name = img_id + '.txt'
    label_path = result_labels_path + '/' + label_file_name
    create_label(label_path, bbox, category_id, img_size)


def create_new_img_wrapper(args):
    """
    Wrapper function for creating new images with parallel processing.

    Args:
        args (tuple): Tuple containing the input arguments for creating a new image.
    """
    image_path, target_img_path, result_images_path, result_labels_path, seg, cat_id = args
    create_new_img(image_path, target_img_path,
                   result_images_path, result_labels_path, seg, cat_id)


def main(dataset_path,
         background_dataset_path,
         result_dataset_path, csv_path):
    """
    Main function to create artificial images and labels.

    Args:
        dataset_path (str): Path to the dataset.
        background_dataset_path (str): Path to the background dataset.
        result_dataset_path (str): Path to save the resulting dataset.
        csv_path (str): Path to the csv file that contains the number of objects in each class in the Plastic Origins dataset.
    """

    sns.set()

    # the different labels and the needed number of annotations for each of them.
    needed_label_num = get_needed_label_num(csv_path)
    print(needed_label_num)
    target_images = get_background_names(background_dataset_path)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, dataset_path)
    result_dataset_path = os.path.join(current_dir, result_dataset_path)
    anns_file_path = os.path.join(dataset_path, 'annotations.json')
    result_images_path = os.path.join(result_dataset_path, 'images')
    result_labels_path = os.path.join(result_dataset_path, "labels")

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
    cats = {}
    for category_name in categories_map.keys():
        if categories_map[category_name] in cats:
            cats[categories_map[category_name]].append(category_name)
        else:
            cats[categories_map[category_name]] = [category_name]

    print(cats)
    for category in cats:
        cat_id = category[0]
        cat_name = category[1]
        needed_num = needed_label_num[cat_name]
        print(cat_id, cat_name, needed_num)

        pylab.rcParams['figure.figsize'] = (14, 14)

        # Obtain Exif orientation tag code
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        # Loads dataset as a coco object
        coco = COCO(anns_file_path)

        # Get image ids
        imgIds = []
        catIds = coco.getCatIds(catNms=cats[category])
        if catIds:
            # Get all images containing an instance of the chosen category
            imgIds = coco.getImgIds(catIds=catIds)
        else:
            # Get all images containing an instance of the chosen super category
            catIds = coco.getCatIds(supNms=cats[category])
            for catId in catIds:
                imgIds += (coco.getImgIds(catIds=catId))
            imgIds = list(set(imgIds))

        imgs = coco.loadImgs(imgIds)
        picked_imgIds = pick_items(imgs, needed_num, 20)

        print(catIds, len(imgIds), needed_num, len(picked_imgIds))
        tasks = []
        for img in picked_imgIds:
            image_path = dataset_path + '/' + img['file_name']
            target_img = pick_random_element(target_images)

            target_img_path = background_dataset_path + '/' + target_img
            print(image_path, target_img_path)
            # Load mask ids
            annIds = coco.getAnnIds(
                imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns_sel = coco.loadAnns(annIds)

            # Show annotations
            ann = pick_random_element(anns_sel)
            for seg in ann['segmentation']:
                # create_new_img(  image_path, target_img_path, result_images_path, result_labels_path, seg, cat_id)
                tasks.append((image_path, target_img_path,
                             result_images_path, result_labels_path, seg, cat_id))

        # Using multithreading to parallelize create_new_img calls
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(create_new_img_wrapper, tasks)


if __name__ == "__main__":
    # Record the start time
    start_time = time.time()

    parser = argparse.ArgumentParser()

    # Define default paths
    default_dataset_path = '../data_TACO/data'
    default_background_dataset_path = '../extracted_background_images'
    default_result_dataset_path = '../artificial_data'
    default_csv_path = '../ref_images/labels.csv'

    # Add arguments
    parser.add_argument("--dataset_path", type=str,
                        default=default_dataset_path, help="Path to the dataset")
    parser.add_argument("--background_dataset_path", type=str,
                        default=default_background_dataset_path, help="Path to the background dataset")
    parser.add_argument("--result_dataset_path", type=str,
                        default=default_result_dataset_path, help="Path to save the resulting dataset")
    parser.add_argument("--csv_path", type=str,
                        default=default_csv_path, help="Path to the csv file that contains the number of existing objects in each class (use the dataset analysis script)")

    args = parser.parse_args()

    dataset_path = args.dataset_path
    background_dataset_path = args.background_dataset_path
    result_dataset_path = args.result_dataset_path
    csv_path = args.csv_path

    # Call the main function and pass the parameters
    main(dataset_path, background_dataset_path, result_dataset_path, csv_path)

    # Record the end time
    end_time = time.time()

    # Calculate the execution time in seconds
    execution_time = end_time - start_time
    print(f"Script executed in {execution_time:.4f} seconds.")
