import os
from pathlib import Path
import yaml
import time
import psycopg2
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import cv2

from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont, ExifTags


def plot_image_and_bboxes(img, anns, ratio:float):

    """ Plots the image and the bounding box(es) associated to the detected object(s).

    Args:
        img (): Image, from the instance file.
        anns (): Annotations linked to the specified image, from instance file.
        ratio (float): Ratio - most often defined at the (1080/height of the image).
    """
    fig, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(img)

    for ann in anns:
        [bbox_x, bbox_y, bbox_w, bbox_h] = (ratio*np.array(ann['bbox'])).astype(int)
        # Obtains the new coordinates of the bboxes - normalized via the ratio.
        rect = patches.Rectangle((bbox_x, bbox_y), bbox_w, bbox_h, linewidth=2, edgecolor='r', facecolor="none")
        ax.add_patch(rect)

    plt.show()
    # Prints out a 12 * 10 image with bounding box(es).


def image_orientation (image:image):
    """ Function which gives the images that have a specified orientation the same orientation.
        If the image does not have an orientation, the image is not altered.

     Args:
        image (image): Image that is in the path data_directory as well as in the instance json files.

    Returns: Image, with the proper orientation.
            _type_: image
    """
    old_orientation = []
    new_orientation = []
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = image._getexif()

        if exif is not None:
            if exif[orientation] == 3:
                image=image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image=image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image=image.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

    return image


def bbox2yolo(bbox, image_height:int=1080, image_width:int=1080):
    """Function to normalize the representation of the bounding box, such that
    there are in the yolo format (normalized in range [0-1])

    Args:
        bbox (list): Coordinates of the bounding box : x, y, w and h coordiates.
        image_height (int, optional): Height of the image. Defaults to 1080.
        image_width (int, optional): Width of the image. Defaults to 1080.

    Returns: Normalized bounding box coordinates : values between 0-1.
        _type_: array
    """
    bbox = bbox.copy().astype(float) #instead of np.int

    bbox[:, [0, 2]] = bbox[:, [0, 2]]/ image_width
    bbox[:, [1, 3]] = bbox[:, [1, 3]]/ image_height

    bbox[:, [0, 1]] = bbox[:, [0, 1]] + bbox[:, [2, 3]]/2

    return bbox


def process_annotations(anns, ratio, target_h, target_w):
    """ Processes the annotations to match the yolo format
    """
    labels = anns["id_ref_trash_type_fk"].values - 1
    bboxes = anns[["location_x","location_y","width","height"]].values * ratio
    bboxes = bbox2yolo(bboxes, target_h, target_w)
    return labels, bboxes


def build_yolo_annotations_for_images(data_dir, images_dir, df_bboxes,
                                      df_images, limit_data):
    """ Generates the .txt files that are necessary for yolo training. See
    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data for data format

    Args:
        data_dir: path of the root data dir. It should contain an
        images folder with all images.
        raw_annotations, raw_images_info, raw_category_info: extracts from
        the database

    Returns:
        the list of images path that will be used in training
    """
    valid_imagenames = []

    input_img_folder = Path(images_dir)
    data_dir = Path(data_dir)
    list_imgs = sorted(os.listdir(input_img_folder))
    used_imgs = set(df_bboxes["id_ref_images_for_labelling"].values)

    print(f"number of images in images folder: {len(list_imgs)}")
    print(f"number of images referenced in database: {len(df_images)}")
    print(f"number of images with a bbox in database: {len(used_imgs)}")

    if not Path.exists(data_dir / "images"):
        os.mkdir(data_dir / "images")
    if not Path.exists(data_dir / "annotations"):
        os.mkdir(data_dir / "annotations")


    count_exists, count_missing = 0, 0

    for img_id in used_imgs:
        img_name = df_images.loc[img_id]["filename"]
        if Path.exists(input_img_folder / img_name):
            count_exists += 1
            if limit_data > 0 and count_exists > limit_data:
                break
            # various meta information about the image, could be useful
            date_creation  = df_images.loc[img_id]["createdon"]
            view           = df_images.loc[img_id]["view"]
            image_quality  = df_images.loc[img_id]["image_quality"]
            context        = df_images.loc[img_id]["context"]
            date_time_obj = time.strptime(date_creation, '%Y-%m-%d %H:%M:%S.%f')

            image = Image.open(input_img_folder / img_name)

            # in place rotation of the image using Exif data
            image_orientation(image)

            image    = np.array(image)
            h, w     = image.shape[:-1]
            target_h = 1080 # the target height of the image
            ratio    = target_h / h # We get the ratio of the target and the actual height
            target_w = int(ratio*w)
            image    = cv2.resize(image, (target_w, target_h))
            h, w     = image.shape[:-1]

            # getting annotations and converting to yolo
            anns = df_bboxes[df_bboxes["id_ref_images_for_labelling"] == img_id]
            labels, bboxes = process_annotations(anns, ratio, target_h, target_w)
            yolo_strs = [str(cat) + " " + " ".join(bbox.astype(str)) for (cat, bbox) in zip(labels, bboxes)]

            # writing the image and annotation
            img_file_name   = data_dir / "images" / (img_id + ".jpg")
            label_file_name = data_dir / "annotations" / (img_id + ".txt")
            Image.fromarray(image).save(img_file_name)
            with open(label_file_name, 'w') as f:
                f.write('\n'.join(yolo_strs))

            valid_imagenames.append(img_file_name.as_posix())
        else:
            count_missing +=1
    return valid_imagenames, count_exists, count_missing

def get_train_valid(list_files, split=0.85):
    """ split data into train and test
    """
    train_files, val_files = train_test_split(list_files, train_size = split)
    train_files = list(set(train_files))
    val_files   = list(set(val_files))

    return train_files, val_files


def generate_yolo_files(output_dir, train_files, val_files):
    """ Generates data files for yolo training: train.txt, val.txt and data.yaml
    """
    output_dir = Path(output_dir)
    with open(output_dir / 'train.txt', 'w') as f:
        for path in train_files:
            f.write(path+'\n')

    with open(output_dir / 'val.txt', 'w') as f:
        for path in val_files:
            f.write(path+'\n')

    data = dict(
        path = './../',
        train = (output_dir / 'train.txt').as_posix() ,
        val = (output_dir / 'val.txt').as_posix(),
        nc = 10,
        names = ['Sheet / tarp / plastic bag / fragment', 'Insulating material', 'Bottle-shaped', 'Can-shaped', 'Drum', 'Other packaging', 'Tire', 'Fishing net / cord', 'Easily namable', 'Unclear'],
        )

    with open(output_dir / 'data.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def get_annotations_from_db(password):
    """ Gets the data from the database. Requires that your IP is configured
    in Azure
    """
    # Update connection string information
    host = "pgdb-plastico-prod.postgres.database.azure.com"
    dbname = "plastico-prod"
    user = "reader_user@pgdb-plastico-prod"
    # password = input('Enter password:')
    password = 'SurfReader!'
    sslmode = "require"

    # Construct connection string
    conn_string = f"host={host} user={user} dbname={dbname} password={password} sslmode={sslmode}"
    conn = psycopg2.connect(conn_string)
    print("Connection established")

    # Fetch all rows from table
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM "label".bounding_boxes')
    raw_annotations = cursor.fetchall()

    cursor.execute('SELECT * FROM "label".images_for_labelling')
    raw_images_info = cursor.fetchall()

    #cursor.execute('SELECT * FROM "campaign".trash_type')
    #raw_category_info = cursor.fetchall()
    df_bboxes = pd.DataFrame(raw_annotations,
                             columns=["id","id_creator_fk","createdon","id_ref_trash_type_fk",
                                      "id_ref_images_for_labelling","location_x","location_y",
                                      "width","height"])

    df_images = pd.DataFrame(raw_images_info,
                             columns=["id","id_creator_fk","createdon","filename","view",
                                      "image_quality","context","container_url","blob_name"])
    conn.close()
    return df_bboxes, df_images.set_index("id") #, raw_category_info


def get_annotations_from_files(input_dir, bbox_filename, images_filename):
    """ Get annotations from csv files instead of the database. The files should
    be located in the input_dir folder
    """
    df_bboxes = pd.read_csv(input_dir / bbox_filename)
    df_images = pd.read_csv(input_dir / images_filename).set_index("id")
    return df_bboxes, df_images


def save_annotations_to_files(output_dir, df_bboxes, df_images):
    """ Saves the annotations in csv format
    """
    df_bboxes.to_csv(output_dir / "bbox.csv")
    df_images.to_csv(output_dir / "images.csv")
