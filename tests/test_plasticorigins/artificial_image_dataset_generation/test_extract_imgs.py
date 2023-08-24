import subprocess
import os
import shutil


def test_num_images():

    current_dir = os.path.dirname(os.path.abspath(__file__))

    functions_dir = os.path.join(
        current_dir, "../../../src/plasticorigins/training/artificial_image_dataset_generation")
    script_path = os.path.join(functions_dir, "extract_imgs_from_videos.py")
    video_dir = os.path.join(current_dir, "ressources/videos")
    save_dir = os.path.join(
        current_dir, "ressources/background_images")
    num_images = 1

    # remove the save directory if it does exist
    if shutil.os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    cmd = f'python {script_path} --num_images {num_images} --video_dir_path {video_dir} --save_dir_path {save_dir}'

    # Run the shell command and capture the output
    output = subprocess.check_output(cmd, shell=True)

    files = os.listdir(save_dir)
    # Check if the folder contains exactly one file
    assert len(files) == 1
    assert os.path.isfile(os.path.join(save_dir, files[0]))
