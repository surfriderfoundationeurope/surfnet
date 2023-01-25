"""The ``yolo`` submodule provides utils functions for YOLO model.

This submodule contains the following classes:

- ``DetectTorchScript`` : Torch script yolo class.

This submodule contains the following functions:

- ``load_model(model_path:str, device:str, conf:float=0.35, iou:float=0.50)`` : Load yolo model from model_path.
- ``predict_yolo(model, img:Mat, size:int=768, augment:bool=False)`` : Interpret yolo prediction object.
- ``voc2centerdims(bboxes:ndarray)`` : Compute center coordinates of the bounding boxes and find their size (width, height).

"""

import json
import cv2
from cv2 import Mat
import numpy as np
from numpy import ndarray, dtype, int64, float64
import torch
import torch.nn as nn
import yaml
import yolov5
from yolov5.models.common import Detections
from yolov5.utils.general import non_max_suppression
from typing import Any, Tuple, Union, Dict, Optional


def load_model(
    model_path: str, device: str, conf: float = 0.35, iou: float = 0.50
) -> Any:

    """Load yolo model from model_path.

    Args:
        model_path (str): the model path
        device (str): type of device used for computing ("cpu", "cuda",...)
        conf (float): confidence value in [0,1]. Set as default to 0.35.
        iou (float): iou value in [0,1]. Set as default to 0.50.

    Returns:
        model (Any): The loaded model with right input parameters
    """

    model = yolov5.load(model_path, device=device)
    model.conf = conf
    model.iou = iou
    model.multi_label = False
    model.max_det = 1000
    categories_id = {v: k for k, v in model.names.items()}
    model.get_id = lambda cat: categories_id[cat]

    return model


def voc2centerdims(bboxes: ndarray) -> ndarray:

    """Compute center coordinates of the bounding boxes and find their size (width, height).

    Args:
        bboxes (ndarray): contain all positions of the bounding boxes : ``voc -> [x1, y1, x2, y2]``

    Returns:
        bboxes (ndarray): the updates bboxes with the center coordinates and (width, height) : ``output -> [x_center, y_center, w, h]``
    """

    bboxes[..., 2:4] -= bboxes[..., 0:2]  # find w,h
    bboxes[..., 0:2] += bboxes[..., 2:4] / 2  # find center

    return bboxes


def predict_yolo(
    model, img: Mat, size: int = 768, augment: bool = False
) -> Tuple[
    ndarray[Any, dtype[int64]],
    ndarray[Any, dtype[float64]],
    ndarray[Any, dtype[int64]],
]:

    """Interpret yolo prediction object.

    Args:
        model (Any): yolo model used for prediction
        img (Mat): image to predict
        size (int): image size. Set as default to 768.
        augment (bool): Data augmentation if True. Set as default to False.

    Returns:
       bboxes (ndarray[Any,dtype[int64]]): The bounding boxes with center coordinates and W, H. Empty array if bboxes is empty.
       confs (ndarray[Any,dtype[float64]]): The confidence values from predictions. Empty array if bboxes is empty.
       labels (ndarray[Any,dtype[int64]]): The list of label ids from predictions. Empty array if bboxes is empty.
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img, size=size, augment=augment)
    preds = results.pandas().xyxy[0]
    bboxes = preds[["xmin", "ymin", "xmax", "ymax"]].values  # voc format

    if len(bboxes):
        bboxes = voc2centerdims(bboxes)
        bboxes = bboxes.astype(int)
        confs = preds.confidence.values
        # Converts back names to ids
        labels = np.array(list(map(model.get_id, preds.name.values)))
        return bboxes, confs, labels

    else:
        return np.array([]), np.array([]), np.array([])


class DetectTorchScript(nn.Module):

    """Torch script yolo class.

    Args:
        weights (ndarray): weights of the object detection model
        conf (float): confidence value. Set as default to ``0.35``.
        iou (float): Set as default to ``0.5``.
        class_names (Optional[Dict]): dictionnary of the object classes for detection in this format : `{class_name : class_id,...}`. Set as default to ``None``.
        data (Optional[str]): path of the yaml file data. Set as default to ``None``.
    """

    max_det = 1000
    agnostic_nms = True
    classes = None

    # YOLOv5 TorchScript class for python inference
    def __init__(
        self,
        weights: ndarray,
        conf: float = 0.35,
        iou: float = 0.50,
        class_names: Optional[Dict] = None,
        data: Optional[str] = None,
    ):
        super().__init__()
        self.class_names = class_names
        self.categories_id = {v: k for k, v in class_names.items()}

        self.conf_thres = conf
        self.iou_thres = iou

        self.size = 640
        stride, names = (
            64,
            [f"class{i}" for i in range(1000)],
        )  # assign defaults

        if data:  # data.yaml path (optional)
            with open(data, errors="ignore") as f:
                names = yaml.safe_load(f)["names"]  # class names

        extra_files = {"config.txt": ""}  # model metadata
        model = torch.jit.load(weights, _extra_files=extra_files)

        if extra_files["config.txt"]:
            d = json.loads(extra_files["config.txt"])  # extra_files dict
            stride, names = int(d["stride"]), d["names"]
        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, image: ndarray) -> Union[torch.Tensor, nn.Module]:

        """Evaluate the yolo model predictions for the input image.

        Args:
            image (ndarray): expects image as numpy array of shape (B x H x W x C) or (H x W x C). In practice B=1 and H x W = 640.

        Returns:
            The model predictions.
        """

        shape = image.shape

        if len(shape) == 3:
            image = np.expand_dims(image, 0)  # add batch axis
        # x = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = letterbox(image, new_shape=self.size, auto=False)[0] # pad and reshape

        x = np.ascontiguousarray(
            image[:, :, :, ::-1].transpose((0, 3, 1, 2))
        )  # BHWC to BCHW, and BGR to RGB
        x = torch.from_numpy(x) / 255  # uint8 to fp16/32

        return self.model(x)[0]

    def get_label_idxs(self, preds_names: ndarray) -> ndarray:

        """Give the class ids knowing the class names.

        Args:
            preds_names (ndarray): list of predicted class names

        Returns:
            The array of the labels ids found in the predictions.
        """

        get_id = lambda cat: self.categories_id[cat]

        return np.array(list(map(get_id, preds_names)))

    def detect(
        self, images: ndarray
    ) -> Tuple[
        ndarray[Any, dtype[int64]],
        ndarray[Any, dtype[float64]],
        ndarray[Any, dtype[int64]],
    ]:

        """Interpret yolo prediction object with batch size set to 1.

        Args:
            images (ndarray): images to predict

        Returns:
            bboxes (ndarray[Any,dtype[int64]]): The bounding boxes with center coordinates and W, H. Empty array if bboxes is empty.
            confs (ndarray[Any,dtype[float64]]): The confidence values from predictions. Empty array if bboxes is empty.
            labels (ndarray[Any,dtype[int64]]): The list of label ids from predictions. Empty array if bboxes is empty.
        """

        pred = self.forward(images)
        pred2 = non_max_suppression(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
        )
        detections = Detections(
            images,
            pred2,
            None,
            times=(0, 0, 0, 0),
            names=self.class_names,
            shape=(1, 3, 640, 640),
        )
        preds = detections.pandas().xyxy[0]
        bboxes = preds[["xmin", "ymin", "xmax", "ymax"]].values  # voc format

        if len(bboxes):
            bboxes = voc2centerdims(bboxes)
            bboxes = bboxes.astype(int)
            confs = preds.confidence.values
            labels = self.get_label_idxs(preds.name.values)
            return bboxes, confs, labels

        else:
            return np.array([]), np.array([]), np.array([])
