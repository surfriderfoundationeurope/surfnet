"""The ``data_processing`` submodule provides several functions for data processing and to build annotations for yolo model.

This submodule contains the following functions :

- ``apply_image_transformations(input_img_folder:WindowsPath, img_name:str)`` : Apply image transformations (orientation, rescaling / resizing).
- ``bbox2yolo(bbox:ndarray, image_height:int=1080, image_width:int=1080)`` : Function to normalize the representation of the bounding box, such that
    there are in the yolo format.
- ``build_bboxes_csv_file_for_DB(
    data_dir: Union[WindowsPath, str],
    images_dir: Union[WindowsPath, str],
    labels_folder_name: Union[str, WindowsPath],
    df_bboxes: DataFrame,
    df_images: DataFrame,
    )`` : Generates the .csv file for updating the DataBase.
- ``build_yolo_annotations_for_images(data_dir:WindowsPath, images_dir:WindowsPath, path_bboxes:str,
                                        df_bboxes:DataFrame, df_images:DataFrame, limit_data:int,
                                         img_folder_name:str, label_folder_name:str, exclude_ids:Optional[set]=None)`` : Generates the .txt files that are necessary for yolo training.
- ``convert_bboxes_to_initial_locations_from_txt_labels(
    labels_folder_path: Union[str, WindowsPath],
    img_id: str,
    target_h: int,
    ratio: float,
    target_w: int,
    mapping_to_10cl: dict,
    )`` : Convert bounding boxes to initial annotation data (location_x, location_y, Width, Height) from .txt label files.
- ``fill_bounding_boxes_table_with_corrections(new_csv_bounding_boxes: Union[WindowsPath, str], user: str, password: str, bboxes_table: str)`` : Fill the bounding boxes DataBase from scratch.
- ``find_img_ids_to_exclude(data_dir:WindowsPath)`` : Find image ids to exclude from list of images used for building the annotation files.
- ``generate_yolo_files(output_dir:WindowsPath, train_files:List[Any,type[str]], val_files:List[Any,type[str]])`` : Generates data files for yolo training: train.txt, val.txt and data.yaml.
- ``get_annotations_from_db(password:str)`` : Gets the data from the database. Requires that your IP is configured in Azure.
- ``get_annotations_from_files(input_dir:WindowsPath, bboxes_filename:str, images_filename:str)`` : Get annotations from csv files instead of the database. The files should be located in the input_dir folder.
- ``get_train_valid(list_files:List[Any,type[str]], split:float=0.85)`` : Split data into train and validation partitions.
- ``image_orientation (image:image)`` : Function which gives the images that have a specified orientation the same orientation.
- ``plot_image_and_bboxes(img:image, anns:list, ratio:float)`` : Plots the image and the bounding box(es) associated to the detected object(s).
- ``process_annotations(anns:DataFrame, ratio:float, target_h:int=1080, target_w:int=1080)`` : Processes the annotations to match the yolo format.
- ``save_annotations_to_files(output_dir:WindowsPath, df_bboxes:DataFrame, df_images:DataFrame)`` : Save the annotations in csv format.
- ``update_bounding_boxes_database(
    data_dir: Union[WindowsPath, str],
    images_dir: Union[WindowsPath, str],
    labels_folder_name: Union[str, WindowsPath],
    new_csv_bounding_boxes: Optional[Union[WindowsPath, str]],
    df_bboxes: DataFrame,
    df_images: DataFrame,
    user: str,
    password: str,
    bboxes_table: str
    )`` : Update directly the Bounding Boxes DataBase from csv file or label folder.

"""

import os
from pathlib import Path, WindowsPath
import yaml
import psycopg2
from typing import Tuple, List, Optional, Union
from datetime import datetime
from tqdm import tqdm

import numpy as np
from numpy import ndarray, array
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame

from matplotlib import image
import matplotlib.pyplot as plt
from PIL import Image, ExifTags, ImageDraw
import cv2


class_id_to_name_mapping = {
    1: "Insulating material",
    4: "Drum",
    2: "Bottle-shaped",
    3: "Can-shaped",
    5: "Other packaging",
    6: "Tire",
    7: "Fishing net / cord",
    8: "Easily namable",
    9: "Unclear",
    0: "Tarp / fragment",
    10: "Sheet",
    11: "Black Tarp / Plastic",
}

mapping_12cl_to_10cl = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "10": 0,
    "11": 0,
}


def plot_image_and_bboxes_yolo(image: image, annotation_list: ndarray) -> None:

    """Plots the image and the bounding box(es) associated to the detected object(s).

    Args:
        image (image): Image, from the instance file
        annotation_list (ndarray): Annotations linked to the specified image, from instance file
    """

    annotations = np.array(annotation_list)

    w, h = image.size

    plotted_image = ImageDraw.Draw(image)

    transformed_annotations = np.copy(annotations)
    transformed_annotations[:, [1, 3]] = annotations[:, [1, 3]] * w
    transformed_annotations[:, [2, 4]] = annotations[:, [2, 4]] * h

    transformed_annotations[:, 1] = transformed_annotations[:, 1] - (
        transformed_annotations[:, 3] / 2
    )
    transformed_annotations[:, 2] = transformed_annotations[:, 2] - (
        transformed_annotations[:, 4] / 2
    )
    transformed_annotations[:, 3] = (
        transformed_annotations[:, 1] + transformed_annotations[:, 3]
    )
    transformed_annotations[:, 4] = (
        transformed_annotations[:, 2] + transformed_annotations[:, 4]
    )

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann

        plotted_image.rectangle(((x0, y0), (x1, y1)), outline="#ff8300", width=5)

        plotted_image.text((x0, y0 - 10), class_id_to_name_mapping[(int(obj_cls))])

    plt.figure(figsize=(12, 10))
    plt.imshow(np.array(image))
    plt.show()


def image_orientation(image: image) -> image:

    """Function which gives the images that have a specified orientation the same orientation.
        If the image does not have an orientation, the image is not altered.

    Args:
        image (image): Image that is in the path data_directory as well as in the instance json files

    Returns:
        image (image): image with the proper orientation
    """

    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == "Orientation":
            break
    exif = image._getexif()

    if exif is not None:
        if orientation == 274:
            return image
        elif exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)

    return image


def bbox2yolo(
    bbox: ndarray, image_height: int = 1080, image_width: int = 1080
) -> ndarray:

    """Function to normalize the representation of the bounding box, such that
    there are in the yolo format (normalized in range [0-1]).

    Args:
        bbox (ndrray): Coordinates of the bounding box : x, y, w and h coordiates
        image_height (int, optional): Height of the image. Set as default to 1080.
        image_width (int, optional): Width of the image. Set as default to 1080.

    Returns:
        bbox (ndarray): Normalized bounding box coordinates : values between 0-1.
    """

    bbox = bbox.copy().astype(float)  # instead of np.int

    bbox[:, [0, 2]] = bbox[:, [0, 2]] / image_width
    bbox[:, [1, 3]] = bbox[:, [1, 3]] / image_height

    bbox[:, [0, 1]] = bbox[:, [0, 1]] + bbox[:, [2, 3]] / 2

    return bbox


def process_annotations(
    anns: DataFrame, ratio: float, target_h: int = 1080, target_w: int = 1080
) -> Tuple[List, ndarray]:

    """Processes the annotations to match the yolo format.

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
    bboxes = anns[["location_x", "location_y", "width", "height"]].values * ratio
    bboxes = bbox2yolo(bboxes, target_h, target_w)
    return labels, bboxes


def apply_filters(
    df_images: DataFrame, context_filters: str, quality_filters: str
) -> DataFrame:

    """Apply context and quality filters if given.

    Args:
        df_images (DataFrame): the dataframe with image informations of context and quality
        context_filters (str): the list of context filters in this format : "[context1,context2,...]". For example, `"[river,nature]"`.
        quality_filters (str): the list of quality filters in this format : "[quality1,quality2,...]". For example, `"[good,medium]"`.

    Returns:
        df_images (DataFrame): the filtered image dataframe
    """

    if context_filters:
        context_filters = context_filters[1:-1].split(",")
        df_images = df_images[df_images["context"].isin(context_filters)]

    if quality_filters:
        quality_filters = quality_filters[1:-1].split(",")
        df_images = df_images[df_images["image_quality"].isin(quality_filters)]

    return df_images


def apply_image_transformations(
    input_img_folder: WindowsPath, img_name: str
) -> Tuple[ndarray, float, int, int]:

    """Apply image transformations (orientation, rescaling / resizing).

    Args:
        input_img_folder (WindowsPath): the image directory
        img_name (str): the image name from df_images

    Returns:
        image (Mat): the resized and transformed image
        ratio (float): the ratio `target_h / h` with `target_h = 1080`
        target_h (int): the target height of the resized image. Set as default to `1080`
        target_w (int): the target weight of the resized image computed as `ratio * w`
    """

    if input_img_folder:
        image = Image.open(input_img_folder / img_name)

    else:
        image = Image.open(img_name)

    # in place rotation of the image using Exif data
    image = image_orientation(image)

    w, h = image.size
    image = np.array(image)
    target_h = 1080  # the target height of the image
    ratio = target_h / h  # We get the ratio of the target and the actual height
    target_w = int(ratio * w)

    image = cv2.resize(image, (target_w, target_h))

    return image, ratio, target_h, target_w


def build_yolo_annotations_for_images(
    data_dir: WindowsPath,
    images_dir: WindowsPath,
    df_bboxes: DataFrame,
    df_images: DataFrame,
    context_filters: str = None,
    quality_filters: str = None,
    limit_data: int = 0,
    exclude_ids: Optional[set] = None,
) -> Tuple[List, int, int]:

    """Generates the .txt files that are necessary for yolo training. See
    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data for data format.

    Args:
        data_dir (WindowsPath): path of the root data directory. It should contain a folder with all useful data for images and annotations.
        images_dir (WindowsPath): path of the image directory. It should contain a folder with all images.
        path_bboxes (str): path of the bounding_boxes csv file
        df_bboxes (DataFrame): DataFrame with the bounding boxes informations (location X, Y and Height, Width)
        df_images (DataFrame): DataFrame with the image informations
        context_filters (str): the list of context filters in this format : "[context1,context2,...]". For example, `"[river,nature]"`. Set as defaults to ``None``.
        quality_filters (str): the list of quality filters in this format : "[quality1,quality2,...]". For example, `"[good,medium]"`. Set as defaults to ``None``.
        limit_data (int): limit number of images used. If you want all images set ``limit_data`` to 0.
        exclude_ids (Optional[set]): Set of image id to exclude from the process. Set as default to ``None``.

    Returns:
        valid_imagenames (List): list of image names that have been processed with success
        cpos (int): number of images with success
        cneg (int): number of images with fail
    """

    valid_imagenames = []
    input_img_folder = Path(images_dir)
    data_dir = Path(data_dir)
    list_imgs = sorted(os.listdir(input_img_folder))
    used_imgs = set(df_bboxes["id_ref_images_for_labelling"].values)

    print(f"number of images in images folder: {len(list_imgs)}")
    print(f"number of images referenced in database: {len(df_images)}")
    print(f"number of images with a bbox in database: {len(used_imgs)}")

    # apply filters if given :
    df_images = apply_filters(df_images, context_filters, quality_filters)

    used_imgs = used_imgs & set(df_images.index)

    print(
        f"number of images after applying context and quality filters: {len(used_imgs)}"
    )

    if exclude_ids:
        used_imgs = used_imgs - exclude_ids
        print(
            f"after exclusion, number of images with a bbox in database: {len(used_imgs)}"
        )

    if not Path.exists(data_dir / "images"):
        os.mkdir(data_dir / "images")
    if not Path.exists(data_dir / "labels"):
        os.mkdir(data_dir / "labels")

    count_exists, count_missing = 0, 0

    print("Start building the annotations ...")

    for img_id in tqdm(used_imgs):

        img_name = df_images.loc[img_id]["filename"]
        if Path.exists(input_img_folder / img_name):
            count_exists += 1
            if limit_data > 0 and count_exists > limit_data:
                break

            image, ratio, target_h, target_w = apply_image_transformations(
                input_img_folder, img_name
            )

            # getting annotations and converting to yolo
            anns = df_bboxes[df_bboxes["id_ref_images_for_labelling"] == img_id]
            labels, bboxes = process_annotations(anns, ratio, target_h, target_w)
            yolo_strs = [
                str(cat) + " " + " ".join(bbox.astype(str))
                for (cat, bbox) in zip(labels, bboxes)
            ]

            # writing the image and annotation
            img_file_name = data_dir / "images" / (img_id + ".jpg")
            label_file_name = data_dir / "labels" / (img_id + ".txt")
            Image.fromarray(image).save(img_file_name)
            with open(label_file_name, "w") as f:
                f.write("\n".join(yolo_strs))

            valid_imagenames.append(img_file_name.as_posix())
        else:
            count_missing += 1

    print(f"Process finished successfully with {count_missing} missing images !")

    return valid_imagenames, count_exists, count_missing


def get_train_valid(
    list_files: List[str], split: float = 0.85
) -> Tuple[List[str], List[str]]:

    """Split data into train and validation partitions.

    Args:
        list_files (List[Any,type[str]]): list of image files to split into train and test partitions
        split (float, optional): train_size between 0 and 1. Set as default to 0.85.

    Returns:
        train_files (List[Any,type[str]]): list of image names for training step
        val_files (List[Any,type[str]]): list of image names for validation step
    """

    train_files, val_files = train_test_split(list_files, train_size=split)
    train_files = list(set(train_files))
    val_files = list(set(val_files))

    return train_files, val_files


def generate_yolo_files(
    output_dir: Union[str, WindowsPath],
    train_files: List[str],
    val_files: List[str],
    nb_classes: int = 10,
) -> None:

    """Generates data files for yolo training: train.txt, val.txt and data.yaml.

    Args:
        output_dir (Union[str,WindowsPath]): path of the root data directory. It should contain a folder with all useful data for images and annotations.
        train_files (List[Any,type[str]]): list of image names for training step
        val_files (List[Any,type[str]]): list of image names for validation step
        nb_classes (int): number of waste classes used for classification
    """

    output_dir = Path(output_dir)

    with open(output_dir / "train.txt", "w") as f:
        for path in train_files:
            f.write(path + "\n")

    with open(output_dir / "val.txt", "w") as f:
        for path in val_files:
            f.write(path + "\n")

    if nb_classes == 12:
        names = [
            "Tarp fragment",
            "Insulating material",
            "Bottle-shaped",
            "Can-shaped",
            "Drum",
            "Other packaging",
            "Tire",
            "Fishing net / cord",
            "Easily namable",
            "Unclear",
            "Sheet",
            "Black Plastic",
        ]

    else:  # nc = 10
        names = [
            "Sheet / tarp / plastic bag / fragment",
            "Insulating material",
            "Bottle-shaped",
            "Can-shaped",
            "Drum",
            "Other packaging",
            "Tire",
            "Fishing net / cord",
            "Easily namable",
            "Unclear",
        ]

    data = dict(
        path="./../",
        train=(output_dir / "train.txt").as_posix(),
        val=(output_dir / "val.txt").as_posix(),
        nc=nb_classes,
        names=names,
    )

    with open(output_dir / "data.yaml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def get_annotations_from_db(
    user: str, password: str, bboxes_table: str
) -> Tuple[DataFrame, DataFrame]:

    """Gets the data from the database. Requires that your IP is configured in Azure.

    Args:
        user (str): username with writing access to the PostgreSql Database
        password (str): password to connect to the SQL DataBase with reading access
        bboxes_table (str): name of the bboxes table from the prod PostgreSql Database

    Returns:
        df_bboxes (DataFrame): DataFrame with the bounding boxes informations (location X, Y and Height, Width)
        df_images (DataFrame): DataFrame with the image informations
    """

    # Update connection string information
    host = "pgdb-plastico-prod.postgres.database.azure.com"
    dbname = "plastico-prod"
    sslmode = "require"

    # Construct connection string
    conn_string = (
        f"host={host} user={user} dbname={dbname} password={password} sslmode={sslmode}"
    )
    conn = psycopg2.connect(conn_string)
    print("Connection established")

    # Fetch all rows from table
    cursor = conn.cursor()

    cursor.execute(f'SELECT * FROM "label".{bboxes_table}')
    raw_annotations = cursor.fetchall()

    cursor.execute('SELECT * FROM "label".images_for_labelling')
    raw_images_info = cursor.fetchall()

    # cursor.execute('SELECT * FROM "campaign".trash_type')
    # raw_category_info = cursor.fetchall()

    df_bboxes = pd.DataFrame(
        raw_annotations,
        columns=[
            "id",
            "id_creator_fk",
            "createdon",
            "id_ref_trash_type_fk",
            "id_ref_images_for_labelling",
            "location_x",
            "location_y",
            "width",
            "height",
        ],
    )

    df_images = pd.DataFrame(
        raw_images_info,
        columns=[
            "id",
            "id_creator_fk",
            "createdon",
            "filename",
            "view",
            "image_quality",
            "context",
            "container_url",
            "blob_name",
        ],
    )
    conn.close()

    return df_bboxes, df_images  # , raw_category_info


def get_annotations_from_files(
    input_dir: WindowsPath, bboxes_filename: str, images_filename: str
) -> Tuple[DataFrame, DataFrame]:

    """Get annotations from csv files instead of the database. The files should be located in the input_dir folder.

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


def save_annotations_to_files(
    output_dir: WindowsPath, df_bboxes: DataFrame, df_images: DataFrame
) -> None:

    """Saves the annotations in csv format.

    Args:
        output_dir (WindowsPath): path of the root data directory. It should contain a folder with all useful data for images and annotations.
        df_bboxes (DataFrame): DataFrame with the bounding boxes informations (location X, Y and Height, Width)
        df_images (DataFrame): DataFrame with the image informations
    """

    df_bboxes.to_csv(output_dir / "bboxes.csv")
    df_images.to_csv(output_dir / "images.csv")


def find_img_ids_to_exclude(data_dir: WindowsPath) -> set:

    """Find image ids to exclude from list of images used for building the annotation files.

    Args:
        data_dir (WindowsPath): path of the root data directory. It should contain a folder with all useful data for images and annotations.

    Returns:
        ids_to_exclude (set): set of image ids to exclude
    """

    list_files = sorted(os.listdir(Path(data_dir) / "labels"))
    ids_to_exclude = {f.split(".")[0] for f in list_files}

    return ids_to_exclude


"""-------- FONCTIONS FOR UPDATING THE DATABASE ---------"""


def convert_bboxes_to_initial_locations_from_txt_labels(
    labels_folder_path: Union[str, WindowsPath],
    img_id: str,
    target_h: int,
    ratio: float,
    target_w: int,
    mapping_to_10cl: dict = None,
) -> Tuple[array, array]:

    """Convert bounding boxes to initial annotation data (location_x, location_y, Width, Height) from .txt label files.

    Args:
        labels_folder_path (Union[str,WindowsPath]): the name of the labels folder or the path od this folder
        img_id (str): the id of the current image
        target_h (int): the target height of the image
        ratio (float): the ratio of the target and the actual height
        target_w (int): the target width of the image
        mapping_to_10cl (dict): dictionary to map categories from nb_classes to 10.

    Returns:
        labels (array[dtype[int64]): the array of the labels presents on the image
        bboxes (array[dtype[int64]): the coordinates of the different bounding boxes of the current image
    """

    labels_folder_path = Path(labels_folder_path)

    with open(labels_folder_path / f"{img_id}.txt") as file:
        lines = file.readlines()

    labels, bboxes = [], []

    for bbox in lines:
        bbox = bbox.split(" ")

        if mapping_to_10cl:
            labels.append(mapping_to_10cl[bbox[0]])
        else:
            labels.append(bbox[0])

        bboxes.append(bbox[1:])

    labels = np.array(labels).astype(int) + 1
    bboxes = np.array(bboxes).astype(float)

    # on décentre la bbox
    bboxes[:, [0, 1]] = bboxes[:, [0, 1]] - bboxes[:, [2, 3]] / 2

    # on dénormalize
    bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * target_w
    bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * target_h

    # dimensions d'origine
    bboxes = bboxes / ratio

    return labels, bboxes.astype(int)


"""--------- UPDATE BOUNDING BOXES DATABASE DIRECTLY ----------"""


def update_bounding_boxes_database(
    data_dir: Union[WindowsPath, str],
    images_dir: Union[WindowsPath, str],
    labels_folder_name: Union[str, WindowsPath],
    new_csv_bounding_boxes: Optional[Union[WindowsPath, str]],
    df_bboxes: DataFrame,
    df_images: DataFrame,
    mapping_to_10cl: dict,
    user: str,
    password: str,
    bboxes_table: str,
) -> None:

    """Update directly the Bounding Boxes DataBase from csv file or label folder. Requires that your IP is configured in Azure.

    Args:
        data_dir (WindowsPath): path of the root data directory. It should contain a folder with all useful data for images and annotations
        images_dir (WindowsPath): path of the image directory. It should contain a folder with all images
        labels_folder_name (Union[str,WindowsPath]): the name of the labels folder or the path od this folder
        new_csv_bounding_boxes (Optional[Union[WindowsPath,str]]) : the path of the bounding boxes csv files with annotation corrections
        df_bboxes (DataFrame): DataFrame with the bounding boxes informations (location X, Y and Height, Width)
        df_images (DataFrame): DataFrame with the image informations
        mapping_to_10cl (dict): dictionary to map categories from nb_classes to 10
        user (str): username with writing access to the PostgreSql Database
        password (str): Password to connect to the Database
        bboxes_table (str): the name of the bounding boxes SQL table
    """

    input_img_folder = Path(images_dir)
    data_dir = Path(data_dir)
    list_imgs = sorted(os.listdir(input_img_folder))
    used_imgs = set(df_bboxes["id_ref_images_for_labelling"].values)
    labels_folder_path = Path(labels_folder_name)

    modified_imgs = set(os.listdir(labels_folder_path))
    modified_ids = {
        img_txt.split(".")[0]
        for img_txt in modified_imgs
        if img_txt.split(".")[0] != ""
    }

    none_modified_imgs = used_imgs - modified_ids

    print(f"number of images in images folder: {len(list_imgs)}")
    print(f"number of images referenced in database: {len(df_images)}")
    print(f"number of images used in database: {len(used_imgs)}")
    print(f"number of images to update in database: {len(modified_ids)}")
    print(
        f"number of images to keep without updates in database: {len(none_modified_imgs)}"
    )

    count_exists = 0

    print("Start updating the annotations ...")

    # Update connection string information
    host = "pgdb-plastico-prod.postgres.database.azure.com"
    dbname = "plastico-prod"
    sslmode = "require"

    # Construct connection string
    conn_string = (
        f"host={host} user={user} dbname={dbname} password={password} sslmode={sslmode}"
    )
    conn = psycopg2.connect(conn_string)
    print("Connection established")

    # Insert all row from csv file
    cursor = conn.cursor()

    if new_csv_bounding_boxes:
        new_df_bboxes = pd.read_csv(new_csv_bounding_boxes)

        for i in tqdm(range(len(new_df_bboxes))):

            row = tuple(new_df_bboxes.loc[i])
            row = tuple(
                [
                    int(val) if (type(val) != str and val is not None) else val
                    for val in row
                ]
            )

            if row[0] is None:

                cursor.execute(
                    f'INSERT INTO "label".{bboxes_table}(id, id_creator_fk, \
                    createdon, id_ref_trash_type_fk, id_ref_images_for_labelling, location_x, location_y, \
                        width, height) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)',
                    row,
                )

            else:

                row = row[1:] + (row[0],)
                cursor.execute(
                    f'UPDATE "label".{bboxes_table} SET (id_creator_fk, \
                    createdon, id_ref_trash_type_fk, id_ref_images_for_labelling, location_x, location_y, \
                        width, height) = (%s, %s, %s, %s, %s, %s, %s, %s) WHERE id = %s',
                    row,
                )

            count_exists += 1
            conn.commit()

    # if we have the label folder name as input
    else:
        for img_id in tqdm(modified_ids):

            img_name = df_images.loc[img_id]["filename"]

            infos_df_bboxes = df_bboxes[
                df_bboxes["id_ref_images_for_labelling"] == img_id
            ]

            nb_trashs = len(infos_df_bboxes)

            _, ratio, target_h, target_w = apply_image_transformations(
                input_img_folder, img_name
            )

            labels, bboxes = convert_bboxes_to_initial_locations_from_txt_labels(
                labels_folder_path, img_id, target_h, ratio, target_w, mapping_to_10cl
            )

            row_diff = nb_trashs - len(labels)

            if row_diff < 0:

                for i in range(nb_trashs):
                    row = (
                        infos_df_bboxes.iloc[i]["id"],
                        int(labels[i]),
                        img_id,
                        int(bboxes[i, 0]),
                        int(bboxes[i, 1]),
                        int(bboxes[i, 2]),
                        int(bboxes[i, 3]),
                    )
                    row = row[1:] + (row[0],)
                    cursor.execute(
                        f'UPDATE "label".{bboxes_table} SET (id_ref_trash_type_fk, id_ref_images_for_labelling, location_x, location_y, \
                                    width, height) = (%s, %s, %s, %s, %s, %s) WHERE id = %s',
                        row,
                    )

                # add new rows for new trashs:
                for i in range(nb_trashs, nb_trashs + row_diff):
                    row = (
                        None,
                        None,
                        datetime.now(),
                        int(labels[i]),
                        img_id,
                        int(bboxes[i, 0]),
                        int(bboxes[i, 1]),
                        int(bboxes[i, 2]),
                        int(bboxes[i, 3]),
                    )
                    cursor.execute(
                        f'INSERT INTO "label".{bboxes_table}(id, id_creator_fk, \
                                    createdon, id_ref_trash_type_fk, id_ref_images_for_labelling, location_x, location_y, \
                                    width, height) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)',
                        row,
                    )

            elif row_diff > 0:

                for i in range(len(labels)):
                    row = (
                        infos_df_bboxes.iloc[i]["id"],
                        int(labels[i]),
                        img_id,
                        int(bboxes[i, 0]),
                        int(bboxes[i, 1]),
                        int(bboxes[i, 2]),
                        int(bboxes[i, 3]),
                    )
                    row = row[1:] + (row[0],)
                    cursor.execute(
                        f'UPDATE "label".{bboxes_table} SET (id_ref_trash_type_fk, id_ref_images_for_labelling, location_x, location_y, \
                                    width, height) = (%s, %s, %s, %s, %s, %s) WHERE id = %s',
                        row,
                    )
            else:

                for i in range(nb_trashs):
                    row = (
                        infos_df_bboxes.iloc[i]["id"],
                        int(labels[i]),
                        img_id,
                        int(bboxes[i, 0]),
                        int(bboxes[i, 1]),
                        int(bboxes[i, 2]),
                        int(bboxes[i, 3]),
                    )
                    row = row[1:] + (row[0],)
                    cursor.execute(
                        f'UPDATE "label".{bboxes_table} SET (id_ref_trash_type_fk, id_ref_images_for_labelling, location_x, location_y, \
                                    width, height) = (%s, %s, %s, %s, %s, %s) WHERE id = %s',
                        row,
                    )

            count_exists += 1
            conn.commit()

    count = cursor.rowcount
    print(count, "successful update into the bounding_boxes table.")

    # Closure of the DB connection
    cursor.close()
    conn.close()
    print("The PostgreSQL connection is close")

    print(f"Process finished successfully with {count_exists} updated images !")


"""--------- UPDATE BOUNDING BOXES DATABASE FROM CSV FILE ----------"""


def build_bboxes_csv_file_for_DB(
    data_dir: Union[WindowsPath, str],
    images_dir: Union[WindowsPath, str],
    labels_folder_name: Union[str, WindowsPath],
    df_bboxes: DataFrame,
    df_images: DataFrame,
    mapping_to_10cl: dict = None,
) -> Tuple[DataFrame, List]:

    """Generates the .csv file for updating the DataBase.

    Args:
        data_dir (WindowsPath): path of the root data directory. It should contain a folder with all useful data for images and annotations.
        images_dir (WindowsPath): path of the image directory. It should contain a folder with all images.
        labels_folder_name (Union[str,WindowsPath]): the name of the labels folder or the path od this folder.
        df_bboxes (DataFrame): DataFrame with the bounding boxes informations (location X, Y and Height, Width)
        df_images (DataFrame): DataFrame with the image informations
        mapping_to_10cl (dict): dictionary to map categories from ``nb_classes`` to ``10``.

    Returns:
        new_df_bboxes (DataFrame): new bounding boxes csv file for initial DataBase including (location X, Y and Height, Width) informations.
        exceptions (List): list of image ids with annotation errors.
    """

    input_img_folder = Path(images_dir)
    data_dir = Path(data_dir)
    list_imgs = sorted(os.listdir(input_img_folder))
    used_imgs = set(df_bboxes["id_ref_images_for_labelling"].values)
    labels_folder_path = Path(labels_folder_name)

    modified_imgs = set(os.listdir(labels_folder_path))
    modified_ids = {
        img_txt.split(".")[0]
        for img_txt in modified_imgs
        if img_txt.split(".")[0] != ""
    }

    none_modified_imgs = used_imgs - modified_ids

    print(f"number of images in images folder: {len(list_imgs)}")
    print(f"number of images referenced in database: {len(df_images)}")
    print(f"number of images used in database: {len(used_imgs)}")
    print(f"number of images to update in database: {len(modified_ids)}")
    print(
        f"number of images to keep without updates in database: {len(none_modified_imgs)}"
    )

    count_exists = 0

    new_df_bboxes = df_bboxes[
        df_bboxes["id_ref_images_for_labelling"].isin(none_modified_imgs)
    ]

    index = 2 * len(new_df_bboxes)

    print("Start updating the annotations ...")

    for img_id in tqdm(modified_ids):

        img_name = df_images.loc[img_id]["filename"]

        infos_df_bboxes = df_bboxes[df_bboxes["id_ref_images_for_labelling"] == img_id]
        nb_trashs = len(infos_df_bboxes)

        _, ratio, target_h, target_w = apply_image_transformations(
            input_img_folder, img_name
        )

        (labels, bboxes,) = convert_bboxes_to_initial_locations_from_txt_labels(
            labels_folder_path, img_id, target_h, ratio, target_w, mapping_to_10cl
        )

        row_diff = nb_trashs - len(labels)

        if row_diff < 0:

            for i in range(nb_trashs):
                index = index + 1
                new_df_bboxes.loc[index] = [
                    infos_df_bboxes.iloc[i]["id"],
                    infos_df_bboxes.iloc[i]["id_creator_fk"],
                    infos_df_bboxes.iloc[i]["createdon"],
                    labels[i],
                    img_id,
                    bboxes[i, 0],
                    bboxes[i, 1],
                    bboxes[i, 2],
                    bboxes[i, 3],
                ]

            index += 20
            # add new rows for new trashs:
            for i in range(nb_trashs, nb_trashs + row_diff):
                index = index + 1
                new_df_bboxes.loc[index] = [
                    None,
                    None,
                    datetime.now(),
                    labels[i],
                    img_id,
                    bboxes[i, 0],
                    bboxes[i, 1],
                    bboxes[i, 2],
                    bboxes[i, 3],
                ]

            index += 20

        elif row_diff > 0:

            for i in range(len(labels)):
                index = index + 1
                new_df_bboxes.loc[index] = [
                    infos_df_bboxes.iloc[i]["id"],
                    infos_df_bboxes.iloc[i]["id_creator_fk"],
                    infos_df_bboxes.iloc[i]["createdon"],
                    labels[i],
                    img_id,
                    bboxes[i, 0],
                    bboxes[i, 1],
                    bboxes[i, 2],
                    bboxes[i, 3],
                ]
            index += 20
        else:

            for i in range(nb_trashs):
                index = index + 1
                new_df_bboxes.loc[index] = [
                    infos_df_bboxes.iloc[i]["id"],
                    infos_df_bboxes.iloc[i]["id_creator_fk"],
                    infos_df_bboxes.iloc[i]["createdon"],
                    labels[i],
                    img_id,
                    bboxes[i, 0],
                    bboxes[i, 1],
                    bboxes[i, 2],
                    bboxes[i, 3],
                ]
            index += 20

        count_exists += 1

    print(f"Process finished successfully with {count_exists} updated images !")

    return new_df_bboxes


def fill_bounding_boxes_table_with_corrections(
    new_csv_bounding_boxes: Union[WindowsPath, str],
    user: str,
    password: str,
    bboxes_table: str,
) -> None:

    """Fill the bounding boxes DataBase from scratch. Requires that your IP is configured in Azure.

    Args:
        new_csv_bounding_boxes (Union[WindowsPath,str]) : the path of the bounding boxes csv files with annotation corrections.
        user (str): username with writing access to the PostgreSql Database
        password (str): password to connect to the PostgreSql Database
        bboxes_table (str): the name of the bounding boxes SQL table
    """

    # Update connection string information
    host = "pgdb-plastico-prod.postgres.database.azure.com"
    dbname = "plastico-prod"
    sslmode = "require"

    # Construct connection string
    conn_string = (
        f"host={host} user={user} dbname={dbname} password={password} sslmode={sslmode}"
    )
    conn = psycopg2.connect(conn_string)
    print("Connection established")

    # Insert all row from csv file
    cursor = conn.cursor()

    new_df_bboxes = pd.read_csv(new_csv_bounding_boxes)

    for i in tqdm(range(len(new_df_bboxes))):

        row = tuple(new_df_bboxes.loc[i])
        row = tuple([int(val) if type(val) != str else val for val in row])

        cursor.execute(
            f'INSERT INTO "label".{bboxes_table}(id, id_creator_fk, \
            createdon, id_ref_trash_type_fk, id_ref_images_for_labelling, location_x, location_y, \
                width, height) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)',
            row,
        )

        conn.commit()
        count = cursor.rowcount
    print(count, "successful inserts into the bounding_boxes table.")

    # Closure of the DB connection
    cursor.close()
    conn.close()
    print("The PostgreSQL connection is close")
