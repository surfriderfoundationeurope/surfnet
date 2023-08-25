from plasticorigins.training.data.make_dataset2 import main  # , args
from argparse import Namespace
import os


PATH = "tests/ressources/"
path_images = PATH + "images2labels/"
path_data = PATH + "data/"

args_1 = Namespace(
    data_dir=path_data,
    images_dir=path_images,
    artificial_data=None,
    bboxes_filename="file_bboxes.csv",
    images_filename="file_images.csv",
    context_filters="[river,nature]",
    quality_filters="[good,medium]",
    nb_classes=12,
    limit_data=0,
    data_augmentation=0,
    exclude_img_folder=path_data + "exclude_ids/",
    split=0.85,
)

args_2 = Namespace(
    data_dir=path_data,
    images_dir=path_images,
    artificial_data=None,
    bboxes_filename=None,
    images_filename=None,
    password=None,
    context_filters=None,
    quality_filters=None,
    nb_classes=12,
    limit_data=0,
    data_augmentation=0,
    exclude_img_folder=None,
    split=0.85,
)

args_3 = Namespace(
    data_dir=path_data,
    images_dir=path_images,
    artificial_data=None,
    bboxes_filename="file_bboxes.csv",
    images_filename="file_images.csv",
    context_filters="[river,nature]",
    quality_filters="[good,medium]",
    nb_classes=12,
    limit_data=0,
    data_augmentation=0,
    exclude_img_folder=None,
    split=0.85,
)


def test_main():

    # valid arguments with csv file to get annotations and exclude ids
    main(args_1)
    os.remove(path_data + "train.txt")
    os.remove(path_data + "val.txt")
    os.remove(path_data + "data.yaml")

    # no valid arguments with csv file to get annotations
    main(args_2)

    # valid arguments with csv file to get annotations without exclude ids
    main(args_3)
    os.remove(path_data + "train.txt")
    os.remove(path_data + "val.txt")
    os.remove(path_data + "data.yaml")
