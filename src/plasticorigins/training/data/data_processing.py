"""The ``data_processing`` submodule provides several functions for data processing and to build annotations for yolo model.

This submodule contains the following functions :

- ``bbox2yolo(bbox:ndarray, image_height:int=1080, image_width:int=1080)`` : Function to normalize the representation of the bounding box, such that
    there are in the yolo format.
- ``build_yolo_annotations_for_images(data_dir:WindowsPath, images_dir:WindowsPath, path_bboxes:str, 
                                        df_bboxes:DataFrame, df_images:DataFrame, limit_data:int,
                                         img_folder_name:str, label_folder_name:str, exclude_ids:Optional[set]=None)`` : Generates the .txt files that are necessary for yolo training.
- ``find_img_ids_to_exclude(data_dir:WindowsPath)`` : Find image ids to exclude from list of images used for building the annotation files.
- ``generate_yolo_files(output_dir:WindowsPath, train_files:List[Any,type[str]], val_files:List[Any,type[str]])`` : Generates data files for yolo training: train.txt, val.txt and data.yaml.
- ``get_annotations_from_db(password:str)`` : Gets the data from the database. Requires that your IP is configured in Azure.
- ``get_annotations_from_files(input_dir:WindowsPath, bboxes_filename:str, images_filename:str)`` : Get annotations from csv files instead of the database. The files should be located in the input_dir folder.
- ``get_train_valid(list_files:List[Any,type[str]], split:float=0.85)`` : Split data into train and validation partitions.
- ``image_orientation (image:image)`` : Function which gives the images that have a specified orientation the same orientation.
- ``plot_image_and_bboxes(img:image, anns:list, ratio:float)`` : Plots the image and the bounding box(es) associated to the detected object(s).
- ``process_annotations(anns:DataFrame, ratio:float, target_h:int=1080, target_w:int=1080)`` : Processes the annotations to match the yolo format.
- ``save_annotations_to_files(output_dir:WindowsPath, df_bboxes:DataFrame, df_images:DataFrame)`` : Save the annotations in csv format.

"""

import os
from pathlib import Path, WindowsPath
import yaml
import psycopg2
from typing import Any, Tuple, List, Optional

import numpy as np
from numpy import ndarray
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame
import cv2

from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ExifTags


def plot_image_and_bboxes(img:image, anns:list, ratio:float) -> None:

    """ Plots the image and the bounding box(es) associated to the detected object(s).

    Args:
        img (image): Image, from the instance file
        anns (list): Annotations linked to the specified image, from instance file
        ratio (float): Ratio - most often defined at the (1080/height of the image)
    """

    _, ax = plt.subplots(1, figsize=(12, 10))
    ax.imshow(img)

    for ann in anns:
        [bbox_x, bbox_y, bbox_w, bbox_h] = (ratio*np.array(ann['bbox'])).astype(int)
        # Obtains the new coordinates of the bboxes - normalized via the ratio.
        rect = patches.Rectangle((bbox_x, bbox_y), bbox_w, bbox_h, linewidth=2, edgecolor='r', facecolor="none")
        ax.add_patch(rect)

    plt.show()
    # Prints out a 12 * 10 image with bounding box(es).


def image_orientation (image:image) -> image:

    """ Function which gives the images that have a specified orientation the same orientation.
        If the image does not have an orientation, the image is not altered.

    Args:
        image (image): Image that is in the path data_directory as well as in the instance json files

    Returns: 
        image (image): image with the proper orientation
    """
    
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

    return image


def bbox2yolo(bbox:ndarray, image_height:int=1080, image_width:int=1080) -> ndarray:

    """Function to normalize the representation of the bounding box, such that
    there are in the yolo format (normalized in range [0-1]).

    Args:
        bbox (ndrray): Coordinates of the bounding box : x, y, w and h coordiates
        image_height (int, optional): Height of the image. Set as default to 1080.
        image_width (int, optional): Width of the image. Set as default to 1080.

    Returns: 
        bbox (ndarray): Normalized bounding box coordinates : values between 0-1.
    """

    bbox = bbox.copy().astype(float) #instead of np.int

    bbox[:, [0, 2]] = bbox[:, [0, 2]]/ image_width
    bbox[:, [1, 3]] = bbox[:, [1, 3]]/ image_height

    bbox[:, [0, 1]] = bbox[:, [0, 1]] + bbox[:, [2, 3]]/2

    return bbox


def process_annotations(anns:DataFrame, ratio:float, target_h:int=1080, target_w:int=1080) -> Tuple[List,ndarray]:

    """ Processes the annotations to match the yolo format.

    Args:
        anns (DataFrame): image annotation informations with image id_ref and coordinates of the bounding box : x, y, w and h coordiates
        ratio (float): Ratio - most often defined at the (1080/height of the image)
        target_h (int): Height of the image. Set as default to ``1080``.
        target_w (int): Width of the image. Set as default to ``1080``.

    Returns: 
        labels (List): list of object classes in the current image
        bboxes (ndarray): array of bounding boxes (x and y positions with height and width) for each object in the current image
    """

    labels = anns["id_ref_trash_type_fk"].values - 1
    bboxes = anns[["location_x","location_y","width","height"]].values * ratio
    bboxes = bbox2yolo(bboxes, target_h, target_w)
    return labels, bboxes


def build_yolo_annotations_for_images(data_dir:WindowsPath, images_dir:WindowsPath, path_bboxes:str, 
                                        df_bboxes:DataFrame, df_images:DataFrame, limit_data:int,
                                         img_folder_name:str, label_folder_name:str, exclude_ids:Optional[set]=None) -> Tuple[List, int, int]:
    
    """ Generates the .txt files that are necessary for yolo training. See
    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data for data format.

    Args:
        data_dir (WindowsPath): path of the root data directory. It should contain a folder with all useful data for images and annotations.
        images_dir (WindowsPath): path of the image directory. It should contain a folder with all images.
        path_bboxes (str): path of the bounding_boxes csv file
        df_bboxes (DataFrame): DataFrame with the bounding boxes informations (location X, Y and Height, Width)
        df_images (DataFrame): DataFrame with the image informations
        limit_data (int): limit number of images used. If you want all images set ``limit_data`` to 0.
        img_folder_name (str): name of the image folder where all annoted images will be stored
        label_folder_name (str): name of the annotation folder where all annotations and labels will be stored
        exclude_ids (Optional[set]): Set of image id to exclude from the process. Set as default to ``None``.

    Returns:
        valid_imagenames (List): list of image names that have been processed with success
        cpos (int): number of images with success
        cneg (int): number of images with fail
    """

    valid_imagenames = []

    list_imgs = sorted(os.listdir(images_dir))
    used_imgs = set(df_bboxes["id_ref_images_for_labelling"].values)

    print(f"number of images in images folder: {len(list_imgs)}")
    print(f"number of images referenced in database: {len(df_images)}")
    print(f"number of images with a bbox in database: {len(used_imgs)}")

    if exclude_ids:
        used_imgs = used_imgs - exclude_ids
        print(f"after exclusion, number of images with a bbox in database: {len(used_imgs)}")

    if not Path.exists(data_dir / img_folder_name):
        os.mkdir(data_dir / img_folder_name)
    if not Path.exists(data_dir / label_folder_name):
        os.mkdir(data_dir / label_folder_name)

    # to determine which methods or annotation processing we will use 
    # (it depends if we work with the whole dataset or not)
    fast_ann_process = False
    if path_bboxes.split("/")[-1].split("_")[0] == "filter":
        fast_ann_process = True

    count_exists, count_missing = 0, 0

    print("Start building the annotations ...")

    for img_id in used_imgs:
        img_name = df_images.loc[img_id]["filename"]
        if Path.exists(images_dir / img_name):
            count_exists += 1
            if limit_data > 0 and count_exists > limit_data:
                break
            
            image = Image.open(images_dir / img_name)

            # in place rotation of the image using Exif data
            try :
                image = image_orientation(image)
            except :
                pass

            image    = np.array(image)
            h, w     = image.shape[:-1]
            target_h = 1080 # the target height of the image
            ratio    = target_h / h # We get the ratio of the target and the actual height
            target_w = int(ratio*w)
            image    = cv2.resize(image, (target_w, target_h))
            h, w     = image.shape[:-1]

            # getting annotations and converting to yolo

            if fast_ann_process:

                bboxes = df_bboxes[df_bboxes["id_ref_images_for_labelling"] == img_id]["bboxes"]
                bboxes = bboxes.iloc[0]
                bboxes = bboxes[1:-1].replace("'","").split(", ")
                yolo_strs = []
                for bbox in bboxes:
                    yolo_strs.append(bbox.strip("\\n"))

            else :

                anns = df_bboxes[df_bboxes["id_ref_images_for_labelling"] == img_id]
                labels, bboxes = process_annotations(anns, ratio, target_h, target_w)
                yolo_strs = [str(cat) + " " + " ".join(bbox.astype(str)) for (cat, bbox) in zip(labels, bboxes)]

            # writing the image and annotation
            img_file_name   = data_dir / img_folder_name / (img_id + ".jpg")
            label_file_name = data_dir / label_folder_name / (img_id + ".txt")
            Image.fromarray(image).save(img_file_name)
            with open(label_file_name, 'w') as f:
                f.write('\n'.join(yolo_strs))

            valid_imagenames.append(img_file_name.as_posix())
        else:
            count_missing +=1

        if count_exists%500==0:
                print("Exists : ", count_exists)
                print("Missing : ",count_missing)

    print(f"Process finished successfully with {count_missing} missing images !")

    return valid_imagenames, count_exists, count_missing


def get_train_valid(list_files:List[Any,type[str]], split:float=0.85) -> Tuple[List[Any,type[str]],List[Any,type[str]]]:

    """Split data into train and validation partitions.

    Args:
        list_files (List[Any,type[str]]): list of image files to split into train and test partitions
        split (float, optional): train_size between 0 and 1. Set as default to 0.85.

    Returns:
        train_files (List[Any,type[str]]): list of image names for training step
        val_files (List[Any,type[str]]): list of image names for validation step
    """

    train_files, val_files = train_test_split(list_files, train_size = split)
    train_files = list(set(train_files))
    val_files   = list(set(val_files))

    return train_files, val_files


def generate_yolo_files(output_dir:WindowsPath, train_files:List[Any,type[str]], val_files:List[Any,type[str]]) -> None:

    """ Generates data files for yolo training: train.txt, val.txt and data.yaml.

    Args:
        output_dir (WindowsPath): path of the root data directory. It should contain a folder with all useful data for images and annotations.
        train_files (List[Any,type[str]]): list of image names for training step
        val_files (List[Any,type[str]]): list of image names for validation step
    """

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


def get_annotations_from_db(password:str) -> Tuple[DataFrame, DataFrame]:

    """ Gets the data from the database. Requires that your IP is configured in Azure.

    Args:
        password (str): password to connect to the SQL DataBase

    Returns:
        df_bboxes (DataFrame): DataFrame with the bounding boxes informations (location X, Y and Height, Width)
        df_images (DataFrame): DataFrame with the image informations
    """

    # Update connection string information
    host = "pgdb-plastico-prod.postgres.database.azure.com"
    dbname = "plastico-prod"
    user = "reader_user@pgdb-plastico-prod"
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


def get_annotations_from_files(input_dir:WindowsPath, bboxes_filename:str, images_filename:str) -> Tuple[DataFrame, DataFrame]:
    
    """ Get annotations from csv files instead of the database. The files should be located in the input_dir folder.

    Args:
        input_dir (WindowsPath): path of the data directory. It should contain a folder with all data (images + annotations).
        bboxes_filename (str): name of the bounding boxes csv file
        images_filename (str): name of the images csv file

    Returns:
        df_bboxes (DataFrame): DataFrame with the bounding boxes informations (location X, Y and Height, Width)
        df_images (DataFrame): DataFrame with the image informations
    """

    df_bboxes = pd.read_csv(input_dir / bboxes_filename)
    df_images = pd.read_csv(input_dir / images_filename).set_index("id")

    return df_bboxes, df_images


def save_annotations_to_files(output_dir:WindowsPath, df_bboxes:DataFrame, df_images:DataFrame) -> None:
    
    """ Saves the annotations in csv format.

    Args:
        output_dir (WindowsPath): path of the root data directory. It should contain a folder with all useful data for images and annotations.
        df_bboxes (DataFrame): DataFrame with the bounding boxes informations (location X, Y and Height, Width)
        df_images (DataFrame): DataFrame with the image informations
    """

    df_bboxes.to_csv(output_dir / "bbox.csv")
    df_images.to_csv(output_dir / "images.csv")


def find_img_ids_to_exclude(data_dir:WindowsPath) -> set:

    """Find image ids to exclude from list of images used for building the annotation files.

    Args:
        data_dir (WindowsPath): path of the root data directory. It should contain a folder with all useful data for images and annotations.

    Returns:
        ids_to_exclude (set): set of image ids to exclude
    """

    list_files = sorted(os.listdir(Path(data_dir) / "labels"))
    ids_to_exclude = set([f.split(".")[0] for f in list_files])

    return ids_to_exclude