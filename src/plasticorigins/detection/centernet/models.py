""" The ``centernet.models`` submodule provides functions to create deep learning model for object detection.

This submodule contains the following functions:

- ``create_model(arch:str, heads:Any, head_conv:Any)`` : Create a model according the input architecture and the heads.

"""

import logging
import torch
from typing import Any
from .networks.mobilenet import get_mobilenet_v3_small

logger = logging.getLogger()


_model_factory = {
    "mobilenetv3small": get_mobilenet_v3_small,
}


def create_model(arch: str, heads: Any, head_conv: Any) -> Any:

    """Create a model according the input architecture and the heads.

    Args:
        arch (str): name of the model architecture
        heads (Any): heads of the model
        head_conv (Any): convolution head

    Returns:
        The model according the input architecture.
    """

    num_layers = int(arch[arch.find("_") + 1 :]) if "_" in arch else 0
    arch = arch[: arch.find("_")] if "_" in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model


def load_model_simple(weights="models/mobilenet_v3_pretrained.pth"):
    model = get_mobilenet_v3_small(num_layers=0, heads={"hm": 1}, head_conv=256)
    checkpoint = torch.load(weights, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model
