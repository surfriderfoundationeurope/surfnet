id_categories = {
    0: "Fragment",  # 'Sheet / tarp / plastic bag / fragment',
    1: "Insulating",  # 'Insulating material',
    2: "Bottle",  # 'Bottle-shaped',
    3: "Can",  # 'Can-shaped',
    4: "Drum",
    5: "Packaging",  # 'Other packaging',
    6: "Tire",
    7: "Fishing net",  # 'Fishing net / cord',
    8: "Easily namable",
    9: "Unclear",
    10: 'Fragment',
    11: 'Fragment'
}


class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


# Old config track with centernet, deprecated
config_track = DotDict(
    {
        "upload_folder": "/tmp",
        "confidence_threshold": 0.004,
        "detection_threshold": 0.3,
        "downsampling_factor": 4,
        "noise_covariances_path": "data/tracking_parameters",
        "output_shape": (960, 544),
        "skip_frames": 3,  # 3
        "arch": "mobilenet_v3_small",
        "device": "cpu",
        "detection_batch_size": 1,
        "display": 0,
        "kappa": 5,  # 7
        "tau": 3,  # 4
        "crop": False,
    }
)

# New config track, the one used in production
config_track_yolo = DotDict(
    {
        "yolo_conf_thrld": 0.35,
        "yolo_iou_thrld": 0.5,
        "upload_folder": "/tmp",
        "confidence_threshold": 0.004,
        "downsampling_factor": 4,
        "noise_covariances_path": "data/tracking_parameters",
        "url_model_yolo": "https://github.com/surfriderfoundationeurope/surfnet/releases/download/v01.2023/yolo_latest.pt",
        "file_model_yolo": "yolov5.pt",
        "output_shape": (768, 768),
        "skip_frames": 3,  # 3
        "arch": "yolo",
        "device": "cpu",
        "detection_batch_size": 1,
        "display": 0,
        "kappa": 5,  # 7
        "tau": 3,  # 4
        "crop": True,
    }
)


logging_config = {
    "version": 1,
    "formatters": {
        "default": {"format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"}
    },
    "handlers": {
        "wsgi": {
            "class": "logging.StreamHandler",
            "stream": "ext://flask.logging.wsgi_errors_stream",
            "formatter": "default",
        },
        "file": {
            "class": "logging.FileHandler",
            "formatter": "default",
            "filename": "errors.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["wsgi", "file"]},
}
