from plasticorigins.training.azure.azure_ml import (
    build_yolo_annotations_for_images_from_azure,
    b64decode_string,
    b64encode_string,
    loadStreamImgToBlobStorage,
)
from plasticorigins.training.data.data_processing import get_annotations_from_db
from argparse import Namespace
import shutil
import os
import numpy as np
from PIL import Image
from azure.storage.blob import BlobServiceClient, ContainerClient
from dotenv import dotenv_values

PATH = "tests/ressources/"

config = dotenv_values(PATH + ".env")


def test_b64encode_string():

    test_msg = "images2label"
    msg_encode = b64encode_string(test_msg)

    if config:
        assert msg_encode == config["input_container_name"]

    else:
        print("EnvError : .env file not found")


def test_b64decode_string():

    if config:
        test_msg = config["input_container_name"]
        msg_decode = b64decode_string(test_msg)

        assert msg_decode == "images2label"

    else:
        print("EnvError : .env file not found")


if config:
    connection_string = b64decode_string(config["connection_string"])
    input_container_name = b64decode_string(config["input_container_name"])
    user_db = b64decode_string(config["user_db"])
    password_db = b64decode_string(config["password_db"])

else:
    connection_string = ""
    input_container_name = ""
    user_db = ""
    password_db = ""


args = Namespace(
    connection_string=connection_string,
    input_container_name=input_container_name,
    user_db=user_db,
    password_db=password_db,
    bboxes_table="bounding_boxes_with_corrections",
    data_dir=PATH,
    context_filters="[river,nature]",
    quality_filters="[good,medium]",
    limit_data=10,
    exclude_img_folder={"3b689af5-7b62-490a-b9c3-b388ec24ebd6"},
)


def test_loadStreamImgToBlobStorage():

    if config:
        blob_service_client = BlobServiceClient.from_connection_string(
            args.connection_string
        )
        container_client = blob_service_client.get_container_client(
            container=args.input_container_name
        )

        blob_list = container_client.list_blobs()

        for blob in blob_list:

            stream = loadStreamImgToBlobStorage(container_client, blob)
            image = np.array(Image.open(stream))
            break

        assert image.shape[:-1] == (3024, 4032)

    else:
        print("EnvError : .env file not found")


def test_build_yolo_annotations_for_images_from_azure():

    if config:

        # Get annotation data from PostgreSql Database
        df_bboxes, df_images = get_annotations_from_db(
            args.user_db, args.password_db, args.bboxes_table
        )

        # without filters and with exclude ids
        valid_imgs, cpos, cneg = build_yolo_annotations_for_images_from_azure(
            args.connection_string,
            args.input_container_name,
            df_bboxes,
            df_images,
            args.data_dir,
            None,
            None,
            args.limit_data,
            args.exclude_img_folder,
        )

        assert os.path.exists(PATH + "images")
        assert os.path.exists(PATH + "labels")

        # remove data folders
        shutil.rmtree(PATH + "images")
        shutil.rmtree(PATH + "labels")

        assert (len(valid_imgs) == 11) and (cpos == 11)

        # with context and quality filters
        valid_imgs, cpos, cneg = build_yolo_annotations_for_images_from_azure(
            args.connection_string,
            args.input_container_name,
            df_bboxes,
            df_images,
            args.data_dir,
            args.context_filters,
            args.quality_filters,
            args.limit_data,
        )

        assert os.path.exists(PATH + "images")
        assert os.path.exists(PATH + "labels")

        # remove data folders
        shutil.rmtree(PATH + "images")
        shutil.rmtree(PATH + "labels")

        images_container_client = ContainerClient.from_connection_string(
            conn_str=args.connection_string, container_name="images"
        )

        labels_container_client = ContainerClient.from_connection_string(
            conn_str=args.connection_string, container_name="labels"
        )

        assert images_container_client.exists()
        assert labels_container_client.exists()

        # remove blob containers
        images_container_client.delete_container()
        labels_container_client.delete_container()

        assert (len(valid_imgs) == 11) and (cpos == 11)

    else:
        print("EnvError : .env file not found")
