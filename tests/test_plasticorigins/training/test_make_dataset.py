import argparse
from plasticorigins.training.data.make_dataset2 import main #, args
from argparse import Namespace


PATH = "tests/ressources/"
path_images = PATH + "images2labels/"
path_data = PATH + "data/"

args_1 = Namespace(
    data_dir = path_data,
    images_dir = path_images,
    bboxes_filename = "file_bboxes.csv",
    images_filename = "file_images.csv",
    context_filters = '[river,nature]',
    quality_filters = '[good,medium]',
    limit_data = 0,
    exclude_img_folder = path_data + "exclude_ids/",
    split = 0.85)

args_2 = Namespace(
    data_dir = path_data,
    images_dir = path_images,
    bboxes_filename = None,
    images_filename = None,
    password = None
    )

args_3 = Namespace(
    data_dir = path_data,
    images_dir = path_images,
    bboxes_filename = "file_bboxes.csv",
    images_filename = "file_images.csv",
    context_filters = '[river,nature]',
    quality_filters = '[good,medium]',
    limit_data = 0,
    exclude_img_folder = None,
    split = 0.85)

def test_main():

    # valid arguments with csv file to get annotations and exclude ids
    main(args_1)

    # no valid arguments with csv file to get annotations
    main(args_2)

    # valid arguments with csv file to get annotations without exclude ids
    main(args_3)

    #assert type(args) == Namespace


