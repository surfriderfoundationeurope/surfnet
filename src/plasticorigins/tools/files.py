import datetime
import os
import os.path as op
from pathlib import Path
from urllib.request import urlretrieve

import cv2


def create_unique_folder(base_folder, filename):
    """Creates a unique folder based on the filename and timestamp"""
    folder_name = op.splitext(op.basename(filename))[0] + "_out_"
    folder_name += datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    output_dir = op.join(base_folder, folder_name)
    if not op.isdir(output_dir):
        os.mkdir(output_dir)
    return output_dir


def download_from_url(url, filename, folder, logger):
    """
    Download a file and place it in the corresponding folder if it does
    not already exists. Useful for spaces demo, do not remove.
    """
    out_filename = Path(folder) / filename
    if not op.exists(out_filename):
        logger.info("---Downloading file...")
        urlretrieve(url, out_filename.resolve().as_posix())
    else:
        logger.info("---File already downloaded.")
    return out_filename


def load_trash_icons(folder_path):
    """loads all icons using cv2 format and returns a dict class -> opened icon
    Useful for spaces demo, do not remove.
    """
    folder_path = Path(folder_path)
    id_path = {
        "Fragment": folder_path
        / "fragment.png",  # 'Fragment',    #'Sheet / tarp / plastic bag / fragment',
        "Insulating": folder_path
        / "mousse.png",  # 'Insulating',  #'Insulating material',
        "Bottle": folder_path
        / "bouteille.png",  # 'Bottle',      #'Bottle-shaped',
        "Can": folder_path / "briquet.png",  # 'Can',         #'Can-shaped',
        "Drum": folder_path / "contenant.png",  # 'Drum',
        "Packaging": folder_path
        / "emballage.png",  # 'Packaging',   #'Other packaging',
        "Tire": folder_path / "pneu.png",  # 'Tire',
        "Fishing net": folder_path
        / "hamecon.png",  # 'Fishing net', #'Fishing net / cord',
        "Easily namable": folder_path / "chaussure.png",  # 'Easily namable',
        "Unclear": folder_path / "dechet.png",  # 'Unclear'
    }
    out_dict = {}
    for idx, path in id_path.items():
        img = cv2.imread(path.resolve().as_posix(), cv2.IMREAD_UNCHANGED)
        resized_img = cv2.resize(img, (100, 60), interpolation=cv2.INTER_AREA)
        out_dict[idx] = resized_img
    return out_dict
