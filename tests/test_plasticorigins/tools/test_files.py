import os
from pathlib import Path

from plasticorigins.tools.files import create_unique_folder, load_trash_icons


def test_create_unique_folder():

    folder = "tests/ressources"
    filename = "test_file.mp4"
    output_dir = create_unique_folder(base_folder=folder, filename=filename)
    new_folder = Path(output_dir)
 
    assert os.path.exists(new_folder)

    os.rmdir(new_folder)


def test_load_trash_icons():

    folder_path = "tests/ressources/icons/"

    dict_trash_icons = load_trash_icons(folder_path)

    assert type(dict_trash_icons) == dict
    assert len(dict_trash_icons) == 10


