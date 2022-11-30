"""The ``azure_ml`` submodule provides several functions for data processing and to build annotations for Azure ML.

This submodule contains the following class :

- ``KeyVault`` : Key Vault Class for Azure ML to get secrets with environement variables.

This submodule contains the following functions :

- ``b64decode_string(base64_message: str, encoding: str = 'ascii')`` : Decode a string object in base 64.
- ``b64encode_string(message: str, encoding: str = 'ascii')`` : Encode a string object in base 64.
- ``loadStreamImgToBlobStorage(container_client:ContainerClient, blob:Any)`` : Load stream image to a specific container from a blob storage.
- ``build_yolo_annotations_for_images_from_azure(connection_string:str, input_container_name:str, df_bboxes:DataFrame,
                                        df_images:DataFrame, data_dir:str, context_filters:str = None,
                                        quality_filters:str = None, limit_data:int=0, exclude_ids:Optional[set]=None
                                        )``: Generates the .txt files that are necessary for yolo training for Azure ML

"""

from plasticorigins.training.data.data_processing import (
    process_annotations,
    apply_image_transformations,
    apply_filters,
)
from typing import Tuple, Optional, List, Any
from pathlib import Path
from PIL import Image
from pandas import DataFrame
import os

from azure.keyvault.secrets import SecretClient
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from io import BytesIO
import base64


class KeyVault:

    """Key Vault Class for Azure ML to get secrets with environement variables."""

    def __init__(
        self, keyvault_name: str, client_id: str, client_secret: str, tenant_id: str
    ):
        self.name = keyvault_name
        self.client_id = client_id
        self.client_secret = client_secret
        self.tenant_id = tenant_id
        self.client = self.__get_client()

    def __get_client(self):
        credential = ClientSecretCredential(
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            client_secret=self.client_secret,
        )
        client = SecretClient(
            vault_url=f"https://{self.name}.vault.azure.net/",
            credential=credential,
        )
        return client

    def get(self, secret: str):
        return self.client.get_secret(secret).value


def b64encode_string(message: str, encoding: str = "ascii") -> str:

    """Encode a string object in base 64.

    Args:
        message (str): message to encode
        encoding (str): type of encoding. Set as default to ``ascii``

    Returns:
        base64_message (str): the message in base 64
    """

    message_bytes = message.encode(encoding)
    base64_bytes = base64.b64encode(message_bytes)
    base64_message = base64_bytes.decode(encoding)

    return base64_message


def b64decode_string(base64_message: str, encoding: str = "ascii") -> str:

    """Decode a string object in base 64.

    Args:
        base64_message (str): the message in base 64
        encoding (str): type of encoding. Set as default to ``ascii``

    Returns:
        message (str): the decoded message
    """

    base64_bytes = base64_message.encode(encoding)
    message_bytes = base64.b64decode(base64_bytes)
    message = message_bytes.decode(encoding)

    return message


def loadStreamImgToBlobStorage(container_client: ContainerClient, blob: Any) -> BytesIO:

    """Load stream image to a specific container from a blob storage.

    Args:
        container_client (ContainerClient): the client container
        blob (Any): the object blob to load from the client container

    Returns:
        stream (BytesIO): the streaming content of the input blob
    """

    blobClient = container_client.get_blob_client(blob)
    stream = BytesIO()
    downloader = blobClient.download_blob()
    downloader.readinto(stream)

    return stream


def build_yolo_annotations_for_images_from_azure(
    connection_string: str,
    input_container_name: str,
    df_bboxes: DataFrame,
    df_images: DataFrame,
    data_dir: str,
    context_filters: str = None,
    quality_filters: str = None,
    limit_data: int = 0,
    exclude_ids: Optional[set] = None,
) -> Tuple[List, int, int]:

    """Generates the .txt files that are necessary for yolo training for Azure ML. See
    https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data for data format.

    Args:
        connection_string (str): connection string to the blob storage which contains all data and images.
        input_container_name (str): name of the input container which contains the images.
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

    used_imgs = set(df_bboxes["id_ref_images_for_labelling"].values)

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

    data_dir = Path(data_dir)

    if not Path.exists(data_dir / "images"):
        os.mkdir(data_dir / "images")
    if not Path.exists(data_dir / "labels"):
        os.mkdir(data_dir / "labels")

    # Creating the client containers
    images_container_client = ContainerClient.from_connection_string(
        conn_str=connection_string, container_name="images"
    )
    if not (images_container_client.exists()):
        images_container_client.create_container()

    labels_container_client = ContainerClient.from_connection_string(
        conn_str=connection_string, container_name="labels"
    )
    if not (labels_container_client.exists()):
        labels_container_client.create_container()

    list_image_filenames = list(df_images[df_images.index.isin(used_imgs)]["filename"])

    # delete blobs from images container
    image_blob_list = images_container_client.list_blobs()
    image_blob_name_list = [blob.name for blob in list(image_blob_list)]
    images_container_client.delete_blobs(*image_blob_name_list)

    # delete blobs from labels container
    label_blob_list = labels_container_client.list_blobs()
    label_blob_name_list = [blob.name for blob in list(label_blob_list)]
    labels_container_client.delete_blobs(*label_blob_name_list)

    count_exists = 0

    print("Start building the annotations ...")

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(
        container=input_container_name
    )
    blob_list = container_client.list_blobs()

    for blob in blob_list:

        img_filename = blob.name
        if img_filename in list_image_filenames:

            stream = loadStreamImgToBlobStorage(container_client, blob)

            if limit_data > 0 and count_exists > limit_data:
                break

            img_id = df_images[df_images["filename"] == img_filename].index.values[0]

            image, ratio, target_h, target_w = apply_image_transformations(None, stream)

            # getting annotations and converting to yolo
            anns = df_bboxes[df_bboxes["id_ref_images_for_labelling"] == img_id]
            labels, bboxes = process_annotations(anns, ratio, target_h, target_w)
            yolo_strs = [
                str(cat) + " " + " ".join(bbox.astype(str))
                for (cat, bbox) in zip(labels, bboxes)
            ]

            # writing the image and annotation

            img_filename = img_id + ".jpg"
            img_local_filename = os.path.join(data_dir, "images/" + img_filename)
            img_blob = blob_service_client.get_blob_client(
                container="images", blob=img_filename
            )

            Image.fromarray(image).save(img_local_filename)
            with open(img_local_filename, "rb") as f:
                img_blob.upload_blob(f)

            label_filename = img_id + ".txt"
            label_local_filename = os.path.join(data_dir, "labels/" + label_filename)
            label_blob = blob_service_client.get_blob_client(
                container="labels", blob=label_filename
            )

            with open(label_local_filename, "w") as f:
                f.write("\n".join(yolo_strs))
            with open(label_local_filename, "rb") as f:
                label_blob.upload_blob(f)

            valid_imagenames.append(Path("./../" + img_local_filename).as_posix())

            count_exists += 1

        if count_exists % 500 == 0:
            print("Exists : ", count_exists)

    count_missing = len(used_imgs) - count_exists

    print(f"Process finished successfully with {count_missing} missing images !")

    return valid_imagenames, count_exists, count_missing
