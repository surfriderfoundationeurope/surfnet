from plasticorigins.tools.files import create_unique_folder
from pathlib import Path

import os


def test_create_unique_folder():
    folder = "tests/ressources"
    filename = "test_file.mp4"
    new_folder = Path(create_unique_folder(base_folder=folder, filename=filename))

    assert os.path.exists(new_folder)

    os.rmdir(new_folder)
