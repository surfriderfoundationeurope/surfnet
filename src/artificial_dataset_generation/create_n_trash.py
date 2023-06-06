import cv2
import os
import numpy as np
import json
import seaborn as sns
from PIL import Image, ExifTags
from pycocotools.coco import COCO
import random
import pylab
import time
import sys
import hashlib


def generate_identifier(image):
    image_data = cv2.imencode('.jpg', image)[1].tobytes()
    hash_object = hashlib.md5(image_data)
    identifier = hash_object.hexdigest()
    return identifier


def get_background_names(folder_path):
    # Get a list of files in the folder
    file_list = os.listdir(folder_path)
    # Print the list of files
    return file_list


def locations(img_size, resized_bbox):
    min_x, min_y, width, height = resized_bbox
    # The percentage of the shape width or height that can be cropped
    p = 0.25
    per_x = round(p*width)
    x = random.randint(- min_x - per_x, img_size[0] - min_x - width + per_x)

    # The shape should be around the horizon (the middle of the image) -->  positioned between 0.25*img_size[1] and 0.75*img_size[1]
    hor = round(0.3*img_size[1])
    y = random.randint(hor - min_y, img_size[1] - hor - min_y - height)
    # x = - min_x - per_x
    return x, y


def get_scale(bounding_box, img_size):
    max_per_w, max_per_h = 0.15, 0.15
    min_per_w, min_per_h = 0.05, 0.05
    max_per = get_resize_ratio(max_per_w, max_per_h, bounding_box, img_size)
    min_per = get_resize_ratio(min_per_w, min_per_h, bounding_box, img_size)
    scale = round(random.uniform(min_per, max_per), 2)
    return scale


def paste_shape(resized_shape, canvas):
    transparence = random.uniform(0, 0.2)
    height, width, _ = resized_shape.shape
    # Calculate the alpha blend
    non_zero_pixels = np.any(resized_shape != 0, axis=2)
    np_condition = np.repeat(
        non_zero_pixels.reshape((height, width, 1)), 3, axis=2)

    alpha_blend = np.where(np_condition,
                           transparence * canvas + (1 - transparence) *
                           resized_shape,
                           canvas
                           )
    return alpha_blend


def create_img_ann(res_path, img_size, img_id):
    file_name = res_path.split("/")[-1]
    dic = {
        "id": img_id,
        "width": img_size[0],
        "height": img_size[1],
        "file_name": file_name
    }
    return dic


def create_obj_ann(ann_id, category_id, new_contour, area, bbox, iscrowd):
    return {
        "id": ann_id,
        "category_id": category_id,
        "segmentation": get_seg(new_contour),
        "area": area,
        "bbox": [bbox],
        "iscrowd": iscrowd
    }


def get_resize_ratio(per_w, per_h, box, img_size):
    min_x, min_y, width, height = box
    W, H = img_size
    return min(per_w * W / width, per_h * H / height)


def get_seg(contour):
    reshaped_contour = np.reshape(contour, (-1, 2))
    flat_contour = reshaped_contour.flatten().tolist()
    return flat_contour


def get_bbox_from_polygon(polygon):
    # Convert the polygon to an OpenCV contour format
    contour = polygon.reshape((-1, 1, 2)).astype(np.int32)
    # Calculate the bounding rectangle using cv2.boundingRect()
    x, y, width, height = cv2.boundingRect(contour)
    return [x, y, width, height]


def get_bbox_from_contour(contour):
    x, y, width, height = cv2.boundingRect(contour)
    return [x, y, width, height]


def crop_polygon(polygon, img_size):
    w, h = img_size
    # Create a mask for points outside the image boundaries
    mask = (polygon[:, 0] >= 0) & (polygon[:, 0] < w) & (
        polygon[:, 1] >= 0) & (polygon[:, 1] < h)

    # Filter the polygon points using the mask
    polygon = polygon[mask]
    return polygon


def rotate_img(image, polygon):
    # Define the rotation angle in degrees
    angle = np.random.randint(-45, 45)
    #angle = 0
    # Get image height and width
    h, w = image.shape[:2]
    img_size = (w, h)
    # Define rotation center as image center
    center = (w / 2, h / 2)
    #print('angle: ', angle)

    # Define rotation matrix using cv2.getRotationMatrix2D
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # Apply rotation and scaling transformation using cv2.warpAffine
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    # Apply same transformation to polygon using cv2.transform
    rotated_polygon = cv2.transform(
        np.array([polygon]), rotation_matrix)[0]

    rotated_polygon = crop_polygon(rotated_polygon, img_size)
    return rotated_image, rotated_polygon


def scale_img(image, polygon):
    # Get image height and width
    h, w = image.shape[:2]
    # Define rotation center as image center
    img_size = (w, h)
    center = (w/2, h/2)
    bounding_box = get_bbox_from_polygon(polygon)
    scale = get_scale(bounding_box, img_size)
    #scale = 1

    # Define scaling matrix using cv2.getRotationMatrix2D
    scaling_matrix = cv2.getRotationMatrix2D(center, 0, scale)
    # Apply rotation and scaling transformation using cv2.warpAffine
    scaled_image = cv2.warpAffine(image, scaling_matrix, (w, h))
    # Apply same transformation to polygon using cv2.transform
    scaled_polygon = np.round((polygon - center) * scale + center).astype(int)

    scaled_polygon = crop_polygon(scaled_polygon, img_size)
    return scaled_image, scaled_polygon


def shift_img(image, polygon):
    # Get image height and width
    h, w = image.shape[:2]
    # Define rotation center as image center
    img_size = (w, h)

    bounding_box = get_bbox_from_polygon(polygon)
    # Define the shift values
    tx, ty = locations(img_size, bounding_box)
    #tx, ty = 0, 0
    #print(tx, ty)

    shifted_image = cv2.warpAffine(image, np.float32(
        [[1, 0, tx], [0, 1, ty]]), (w, h))
    shifted_polygon = polygon + [tx, ty]

    shifted_polygon = crop_polygon(shifted_polygon, img_size)
    return shifted_image, shifted_polygon


def transform_img(image, polygon):
    # Apply rotation, scale, and shift to the image
    try:
        image, polygon = rotate_img(image, polygon)
        if np.array_equal(polygon, np.array([])):
            raise ValueError(
                "After rotation, polygon is outside of the transformed image")
        image, polygon = scale_img(image, polygon)
        if np.array_equal(polygon, np.array([])):
            raise ValueError(
                "After scaling, polygon is outside of the transformed image")
        image, polygon = shift_img(image, polygon)
        if np.array_equal(polygon, np.array([])):
            raise ValueError(
                "After shifting, polygon is outside of the transformed image")
        return image, polygon

    except Exception as e:
        print(
            "After transformations, polygon is outside of the transformed image: ", str(e))


def add_to_json(ann_path, ann, key):
    # Load the JSON file into a Python dictionary
    with open(ann_path, "r") as f:
        data = json.load(f)

    # Append the new dictionary to the 'points' list
    data[key].append(ann)

    # Write the modified dictionary back to the file as JSON
    with open(ann_path, "w") as f:
        json.dump(data, f)


def create_new_json(categories, ann_path):
    # Create a Python dictionary containing the data
    cat_dic = []
    for x in categories:
        cat_dic.append({
            "id": x[0],
            "name": x[1]})
    data = {
        "images": [],
        "annotations": [],
        "categories": cat_dic,
    }
    # Convert the dictionary to a JSON string
    json_str = json.dumps(data)
    # Open a new file in write mode
    with open(ann_path, "w") as f:
        # Write the JSON string to the file
        f.write(json_str)


def create_img(img_path, seg):
    try:
        # Load image and annotation file
        image = cv2.imread(img_path)

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
        except:
            # Catch any uncaught exception and print its type
            print("An uncaught exception occurred:")
            exc_type, _, _ = sys.exc_info()
            print("Exception type:", exc_type)
        finally:
            # Find contours in mask
            # contours, _ = cv2.findContours( mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # Find the largest contour
            # largest_contour = max(contours, key=cv2.contourArea)
            contour = polygon.reshape((-1, 1, 2)).astype(np.int32)
            box = get_bbox_from_contour(contour)
            area = cv2.contourArea(contour)

            # Paste the contour on the mask
            cv2.drawContours(shape_mask, [contour], 0, 255, -1)
            # Apply the mask to the original image to extract the shape
            shape = cv2.bitwise_and(image, image, mask=shape_mask)
            return shape, contour, box, area

    except Exception as e:
        print(str(e))


def create_new_img(image_path, target_img_path, res_dataset_path, new_ann_path, input_anns, category_id):

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
        try:
            seg = ann['segmentation']
            iscrowd = ann['iscrowd']
            # create the new image, contour, area & bbox
            shape, new_contour, bbox, area = create_img(
                image_path, seg)

            # Paste the shape onto the canvas on top of the target image
            canvas = paste_shape(shape, canvas)

            # create the object annotations
            ann_id = generate_identifier(canvas)
            obj_ann = create_obj_ann(
                ann_id, category_id, new_contour, area, bbox, iscrowd)
            anns.append(obj_ann)
            print(str(ann['id']) + " done")

        except Exception as e:
            print(
                f"An error occurred with {ann['id']} in the image '{image_path}': {e}")
            continue
    # Save the result as a new image
    img_id = generate_identifier(canvas)
    filename = 'image_' + img_id + '.jpg'
    res_path = res_dataset_path + '/' + filename
    cv2.imwrite(res_path, canvas)
    print("drawing image...")

    img_ann = create_img_ann(filename, img_size, img_id)
    add_to_json(new_ann_path, img_ann, "images")

    print("annotations ...")
    # create the image annotations
    # print(anns)
    for ann in anns:
        ann['image_id'] = img_id
        add_to_json(new_ann_path, ann, "annotations")


def main(dataset_path='./data_TACO/data',
         background_dataset_path='./extracted_background_images',
         result_dataset_path='./artificial_data_n_objects'):
    sns.set()

    # Paths
    anns_file_path = dataset_path + '/annotations.json'
    new_ann_path = result_dataset_path + "/annotations.json"

    # create the save directory if it does not exist
    if not os.path.exists(result_dataset_path):
        os.makedirs(result_dataset_path)

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    # Getting information from the json file
    imgs = dataset['images']

    # Mapping the TACO categories to the plastic origins categories
    categories_map = {
        'Bottle': (0, 'Bottle-shaped'),
        'Can': (1, 'Can-shaped'),
        # 'Unlabeled litter': (2, 'Unclear'),
        # 'Scrap metal': (2, 'Unclear'),
        'Rope & strings': (3, 'Fishing Net / Cord'),
        'Carton': (4, 'Other packaging'),
        'Cup': (4, 'Other packaging'),
        'Blister pack': (4, 'Other packaging'),
        'Paper': (4, 'Other packaging'),
        'Plastic container': (4, 'Other packaging'),
        # 'Shoe': (5, 'Easily namable'),
        'Styrofoam piece': (6, 'Insulating material'),
        # 'Plastic bag & wrapper': (7, 'Tarp fragment'),
        'Garbage bag': (8, 'Black Plastic')
    }

    # Creating the annotation json file and putting the basic information inside
    create_new_json(set(categories_map.values()), new_ann_path)

    totalCatIds = []
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
        catIds = coco.getCatIds(catNms=[category_name])
        totalCatIds += catIds
        if not catIds:
            # Get all images containing an instance of the chosen super category
            catIds = coco.getCatIds(supNms=[category_name])
            totalCatIds += catIds

    # Get all images containing an instance of the chosen category
    imgIds = []
    # print(totalCatIds)
    for catId in totalCatIds:
        imgIds += (coco.getImgIds(catIds=catId))

    imgIds = list(set(imgIds))
    imgs = coco.loadImgs(imgIds)
    #print(imgIds)

    for img in imgs:
        image_path = dataset_path + '/' + img['file_name']
        target_images = get_background_names(background_dataset_path)
        for target_img in target_images:
            target_img_path = background_dataset_path + '/' + target_img

            # Load mask ids
            annIds = coco.getAnnIds(
                imgIds=img['id'], catIds=totalCatIds, iscrowd=None)
            anns_sel = coco.loadAnns(annIds)
            #print(annIds)
            create_new_img(image_path, target_img_path, result_dataset_path,
                           new_ann_path, anns_sel, category_id)


if __name__ == "__main__":
    # Paths
    dataset_path = './data_TACO/data'
    background_dataset_path = './extracted_background_images'
    result_dataset_path = './artificial_data_n_objects'

    # Check if the parameters were provided
    if len(sys.argv) > 3:
        result_dataset_path = sys.argv[3]
    if len(sys.argv) > 2:
        background_dataset_path = sys.argv[2]
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]

    # Call the main function and pass the parameter
    main(dataset_path, background_dataset_path, result_dataset_path)
