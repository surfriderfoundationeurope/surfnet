from plasticorigins.training.data.data_processing import (image_orientation, 
                                                                bbox2yolo, 
                                                                process_annotations,
                                                                build_yolo_annotations_for_images,
                                                                generate_yolo_files,
                                                                get_train_valid, 
                                                                get_annotations_from_files,
                                                                get_annotations_from_db,
                                                                save_annotations_to_files,
                                                                find_img_ids_to_exclude,
                                                                convert_bboxes_to_initial_locations_from_txt_labels,
                                                                build_bboxes_csv_file_for_DB,
                                                                plot_image_and_bboxes_yolo
                                                            )
import numpy as np
from PIL import Image, ExifTags
import os
import shutil
import pandas as pd
from pathlib import Path
from argparse import Namespace

PATH = "tests/ressources/"
path_images = PATH + "images2labels/"
path_data = PATH + "data/"
path_inputs = path_data + "inputs/"
path_outputs = path_data + "outputs/"
path_test_images = PATH + "test_images/"

args = Namespace(
    data_dir = path_data,
    images_dir = path_images,
    bboxes_filename = "file_bboxes.csv",
    images_filename = "file_images.csv",
    context_filters = '[river,nature]',
    quality_filters = '[good,medium]',
    limit_data = 0)

df_bboxes = pd.read_csv(args.data_dir + args.bboxes_filename)
df_images = pd.read_csv(args.data_dir + args.images_filename).set_index("id")


def test_image_orientation():

    image = Image.open(path_test_images + "879f4832-991f-455d-85bd-3a49aef7191d(21).jpg")
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation]=='Orientation':
            break
    exif = image._getexif()

    assert (exif[orientation] == 8)

    image_orient = image_orientation(image)
        
    # rotation of 90°
    assert image_orient.size[0] == image.size[1]
    assert image_orient.size[1] == image.size[0]


def test_bbox2yolo():

    img_id = "aa51b47b-3320-4b1a-b199-71e90b495863"
    img_name = df_images.loc[img_id]["filename"]

    image = Image.open(path_images + img_name)
    image = image_orientation(image)
    image    = np.array(image)

    h, w     = image.shape[:-1]
    target_h = 1080 # the target height of the image
    ratio    = target_h / h # We get the ratio of the target and the actual height

    anns = df_bboxes[df_bboxes["id_ref_images_for_labelling"] == img_id]
    bboxes = anns[["location_x","location_y","width","height"]].values * ratio

    assert (bbox2yolo(bboxes, h, w) <= 1).all()


def test_process_annotations():

    img_id = "aa51b47b-3320-4b1a-b199-71e90b495863"
    img_name = df_images.loc[img_id]["filename"]

    image = Image.open(path_images + img_name)
    image = image_orientation(image)
    image    = np.array(image)

    h, w     = image.shape[:-1]
    target_h = 1080 # the target height of the image
    ratio    = target_h / h # We get the ratio of the target and the actual height
    target_w = int(ratio*w)

    anns = df_bboxes[df_bboxes["id_ref_images_for_labelling"] == img_id]

    labels, bboxes = process_annotations(anns, ratio, target_h, target_w)

    assert np.array_equal(labels, np.array([0,0]))
    assert np.array_equal(np.round(bboxes,2), np.array([[0.53, 0.66, 0.08, 0.08], [0.43, 0.76, 0.03, 0.04]]))

def test_build_yolo_annotations_for_images():

    # remove to rerun the test_build_yolo_annotations_for_images
    shutil.rmtree(path_data + "images")
    shutil.rmtree(path_data + "labels")

    # no limit data with exclude ids and no filters
    exclude_ids = {"2247af7a-e86c-48d5-87f9-9cbebba073da"}
    valid_imgs, cpos, cneg = build_yolo_annotations_for_images(args.data_dir, args.images_dir, df_bboxes, df_images, None, None, args.limit_data, exclude_ids)

    assert os.path.exists(path_data + "images")
    assert os.path.exists(path_data + "labels")    
    assert (len(valid_imgs) == 9) and (cpos == 9) and (cneg == 0)

    shutil.rmtree(path_data + "images")
    shutil.rmtree(path_data + "labels")

    # with limit data
    limit_data = 4
    valid_imgs, cpos, cneg = build_yolo_annotations_for_images(args.data_dir, args.images_dir, df_bboxes, df_images, args.context_filters, args.quality_filters, limit_data)

    assert os.path.exists(path_data + "images")
    assert os.path.exists(path_data + "labels")    
    assert (len(valid_imgs) == 4) and (cpos == 5) and (cneg == 0)

    shutil.rmtree(path_data + "images")
    shutil.rmtree(path_data + "labels")

    # no limit data with context and quality filters
    valid_imgs, cpos, cneg = build_yolo_annotations_for_images(args.data_dir, args.images_dir, df_bboxes, df_images, args.context_filters, args.quality_filters, args.limit_data)
    
    assert os.path.exists(path_data + "images")
    assert os.path.exists(path_data + "labels")    
    assert len(os.listdir(path_data + "images")) == len(os.listdir(path_data + "labels")) == 9
    assert (len(valid_imgs) == 9) and (cpos == 9) and (cneg == 0)

def test_get_train_valid():

    list_imgs_test = os.listdir(path_data + "images")

    train_files, val_files = get_train_valid(list_imgs_test,0.85)

    assert (len(train_files) == 7) and (len(val_files) == 2)


def test_generate_yolo_files():

    data_dir = Path(path_data)
    to_exclude = None

    yolo_filelist, _, _ = build_yolo_annotations_for_images(data_dir, args.images_dir,df_bboxes, df_images,
             args.context_filters, args.quality_filters, args.limit_data, to_exclude)

    train_files, val_files = get_train_valid(yolo_filelist, 0.85)
    generate_yolo_files(Path(path_outputs), train_files, val_files)

    assert os.path.exists(path_outputs + 'train.txt')
    assert os.path.exists(path_outputs + 'val.txt')
    assert os.path.exists(path_outputs + 'data.yaml')  


def test_get_annotations_from_files():

    df_bboxes, df_images = get_annotations_from_files(Path(args.data_dir), args.bboxes_filename, args.images_filename)

    assert df_bboxes.shape == (13,9)
    assert df_images.shape == (10,8)


# def test_get_annotations_from_db():

#     df_bboxes, df_images = get_annotations_from_db('SurfReader!')

#     assert df_bboxes.shape == (13,9)
#     assert df_images.shape == (10,8)


def test_save_annotations_to_files():

    save_annotations_to_files(Path(path_outputs), df_bboxes, df_images)

    assert os.path.exists(path_outputs + "bboxes.csv")
    assert os.path.exists(path_outputs + "images.csv")


def test_find_img_ids_to_exclude():

    list_exclude_ids = find_img_ids_to_exclude(Path(path_data + "exclude_ids"))

    assert len(list_exclude_ids) == 3


def test_convert_bboxes_to_initial_locations_from_txt_labels():

    labels_folder_path = path_data + "labels"

    img_id = "aa51b47b-3320-4b1a-b199-71e90b495863"
    img_name = df_images.loc[img_id]["filename"]

    image = Image.open(path_images + img_name)
    image = image_orientation(image)
    image    = np.array(image)

    h, w     = image.shape[:-1]
    target_h = 1080 # the target height of the image
    ratio    = target_h / h # We get the ratio of the target and the actual height
    target_w = int(ratio*w)

    labels, bboxes = convert_bboxes_to_initial_locations_from_txt_labels(labels_folder_path, img_id, target_h, ratio, target_w)

    assert np.array_equal(labels, np.array([1, 1]))
    assert np.array_equal(bboxes, np.array([[2932,2479,455,312],[2504,2975,197,169]]))


def test_build_bboxes_csv_file_for_DB():

    images_dir = path_images
    labels_dir = path_data + "labels_modif/"
    new_df_bboxes, exceptions = build_bboxes_csv_file_for_DB(args.data_dir, images_dir, labels_dir, df_bboxes, df_images)

    assert new_df_bboxes.shape == (11,9) 
    assert len(exceptions) == 2


def test_plot_image_and_bboxes():

    img_id = "aa51b47b-3320-4b1a-b199-71e90b495863"
    img_label = img_id + ".txt"

    with open(Path(path_data + "labels") / img_label) as file:
        lines = file.readlines()

    bboxes = []
    for line in lines:
        line = line.split()
        bboxes.append([int(line[0])] + list(map(float,line[1:])))

    image = Image.open(path_data + "images/" + img_id + ".jpg")
    image = image_orientation(image)

    plot_image_and_bboxes_yolo(image, bboxes)
