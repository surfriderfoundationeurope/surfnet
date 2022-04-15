import json

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
import yolov5
from yolov5.models.common import Detections
from yolov5.utils.general import non_max_suppression

# This has to be kept for now as this depends on the model training
id_categories = {
    1: "Insulating material",
    4: "Drum",
    2: "Bottle-shaped",
    3: "Can-shaped",
    5: "Other packaging",
    6: "Tire",
    7: "Fishing net / cord",
    8: "Easily namable",
    9: "Unclear",
    0: "Sheet / tarp / plastic bag / fragment",
}

categories_id = {v: k for k, v in id_categories.items()}
get_id = lambda cat: categories_id[cat]


def load_model(model_path, device, conf=0.35, iou=0.50):
    model = yolov5.load(model_path, device=device)
    model.conf = conf
    model.iou = iou
    model.classes = None
    model.multi_label = False
    model.max_det = 1000
    return model


def voc2centerdims(bboxes):
    """
    voc  => [x1, y1, x2, y2]
    output => [xcenter, ycenter, w, h]
    """
    bboxes[..., 2:4] -= bboxes[..., 0:2]  # find w,h
    bboxes[..., 0:2] += bboxes[..., 2:4] / 2  # find center
    return bboxes


def predict_yolo(model, img, size=768, cvtColor=True, augment=False):
    """
    interpret yolo prediction object
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img, size=size, augment=augment)
    preds = results.pandas().xyxy[0]
    bboxes = preds[["xmin", "ymin", "xmax", "ymax"]].values  # voc format
    if len(bboxes):
        bboxes = voc2centerdims(bboxes)
        bboxes = bboxes.astype(int)
        confs = preds.confidence.values
        labels = np.array(list(map(get_id, preds.name.values)))
        return bboxes, confs, labels
    else:
        return np.array([]), np.array([]), np.array([])


class DetectTorchScript(nn.Module):
    """Torch script yolo class"""

    max_det = 1000
    agnostic_nms = True
    classes = None

    # YOLOv5 TorchScript class for python inference
    def __init__(
        self,
        weights,
        device=None,
        conf=0.35,
        iou=0.50,
        class_names=None,
        dnn=False,
        data=None,
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

    def forward(self, im):
        """expects im as np array of shape BHWC or HWC
        in practice B=1
        HW=size=640
        """
        shape = im.shape
        if len(shape) == 3:
            im = np.expand_dims(im, 0)  # add batch axis
        # x = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # im = letterbox(im, new_shape=self.size, auto=False)[0] # pad and reshape
        x = np.ascontiguousarray(
            im[:, :, :, ::-1].transpose((0, 3, 1, 2))
        )  # BHWC to BCHW, and BGR to RGB
        x = torch.from_numpy(x) / 255  # uint8 to fp16/32
        return self.model(x)[0]

    def get_label_idxs(self, preds_names):
        get_id = lambda cat: self.categories_id[cat]
        return np.array(list(map(get_id, preds_names)))

    def detect(self, ims):
        """Assumes batch size = 1 for now"""
        pred = self.forward(ims)
        pred2 = non_max_suppression(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes,
            agnostic=self.agnostic_nms,
            max_det=self.max_det,
        )
        detections = Detections(
            ims,
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
