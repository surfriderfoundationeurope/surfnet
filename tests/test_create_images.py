import subprocess
import os
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
functions_dir = os.path.join(current_dir, "../src/artificial_dataset_generation")
script_path = os.path.join(functions_dir, "create_images.py")
dataset_path = os.path.join(current_dir, "ressources/data")
background_dataset_path = os.path.join(
    current_dir, "ressources/background_images")
result_dataset_path = os.path.join(
    current_dir, "ressources/artificial_data")
result_images_path = os.path.join(
    result_dataset_path, "images")
result_labels_path = os.path.join(
    result_dataset_path, "labels")

def test_num_images():
    # remove the save directory if it does exist
    if shutil.os.path.exists(result_dataset_path):
        shutil.rmtree(result_dataset_path)

    cmd = f'python {script_path} --dataset_path {dataset_path} --background_dataset_path {background_dataset_path} --result_dataset_path {result_dataset_path}'

    # Run the shell command and capture the output
    output = subprocess.check_output(cmd, shell=True)

    images = os.listdir(result_images_path)

    assert len(images) == 4
    for file in images:
        assert os.path.isfile(os.path.join(result_images_path, file))
        file_extension = os.path.splitext(file)[1].lower()
        assert file_extension == '.jpg'

def test_num_labels():
    # remove the save directory if it does exist
    if shutil.os.path.exists(result_dataset_path):
        shutil.rmtree(result_dataset_path)

    cmd = f'python {script_path} --dataset_path {dataset_path} --background_dataset_path {background_dataset_path} --result_dataset_path {result_dataset_path}'

    # Run the shell command and capture the output
    output = subprocess.check_output(cmd, shell=True)

    files = sorted(os.listdir(result_labels_path))

    assert len(files) == 4
    for file in files:
        assert os.path.isfile(os.path.join(result_labels_path, file))
        file_extension = os.path.splitext(file)[1].lower()
        assert file_extension == '.txt'


def test_label_components():
    # remove the save directory if it does exist
    if shutil.os.path.exists(result_dataset_path):
        shutil.rmtree(result_dataset_path)

    cmd = f'python {script_path} --dataset_path {dataset_path} --background_dataset_path {background_dataset_path} --result_dataset_path {result_dataset_path}'

    # Run the shell command and capture the output
    output = subprocess.check_output(cmd, shell=True)

    files = sorted(os.listdir(result_labels_path))
    for file in files:
        file_path = os.path.join(result_labels_path, file)

        with open(file_path, 'r') as f:
            line_count = sum(1 for _ in f)
            assert line_count == 1