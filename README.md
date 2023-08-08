# Project README

This README provides an overview of the various scripts within this project and their respective functionalities. Please refer to this guide for a clear understanding of how to use each script effectively.

## Files and Descriptions

### categories_map.py

This file facilitates the mapping between the TACO and PlasticOrigins classes, enhancing data compatibility.

### utils.py

Containing essential functions, this script supports `create_images.py`, `create_images_num.py`, and `create_n_trash.py` by providing necessary functionalities.

### create_images.py

Artificially generates images for object detection by combining TACO dataset objects and background images. It creates a specified number of images with each object pasted on a unique background.

**Input:**
- `--dataset_path`: Path to the TACO dataset
- `--background_dataset_path`: Path to the background images dataset
- `--result_dataset_path`: Path to save the resulting dataset
- `--num_uses_background`: Number of background images used per object

### create_images_num.py

Similar to `create_images.py`, this script artificially generates images for object detection. However, it aims to achieve a more balanced distribution of objects among different classes in the resulting dataset.

**Input:**
- `--dataset_path`: Path to the TACO dataset
- `--background_dataset_path`: Path to the background images dataset
- `--result_dataset_path`: Path to save the resulting dataset
- `--csv_path`: Path to the CSV file containing existing object counts per class

### create_n_trash.py

Generates images for object detection by placing all objects from a TACO image onto a single background image.

**Input:**
- `--dataset_path`: Path to the TACO dataset
- `--background_dataset_path`: Path to the background images dataset
- `--result_dataset_path`: Path to save the resulting dataset

### dataset_analysis.py

Creates CSV and bar chart files indicating the number of objects in each class. 

**Input:**
- `--folder_path`: Path to the folder containing image labels
- `--csv_labels_file_name`: Name of the CSV labels file to be generated (default: labels.csv)
- `--png_labels_file_name`: Name of the PNG labels file to be generated (default: labels.png)
- `--results_path`: Path to save the graph and CSV

### dataset_analysis_train.py

Performs the same function as `dataset_analysis.py`.

**Input:**
- `--train_path`: Path to the train.txt file for extracting label files
- `--val_path`: Path to the val.txt file for extracting label files (optional)
- `--csv_labels_file_name`: Name of the CSV labels file to be generated (default: labels.csv)
- `--png_labels_file_name`: Name of the PNG labels file to be generated (default: labels.png)
- `--results_path`: Path to save the graph and CSV

### draw_bbox.py

Draws bounding boxes for objects on an image.

**Input:**
- `--image_path`: Path to the image
- `--annotation_path`: Path to the annotation file (format: class_id, center_x, center_y, w, h)

### draw_bbox_taco.py

Similar to `draw_bbox.py`, but uses JSON annotations.

**Input:**
- `--image_path`: Path to the image
- `--annotation_path`: Path to the JSON annotation (format: bbox: min_x, min_y, w, h in pixels)

### draw_polygons.py

Draws object segments on an image using JSON annotations.

**Input:**
- `--image_path`: Path to the image
- `--annotation_path`: Path to the JSON annotation (format: bbox: min_x, min_y, w, h in pixels)

### extract_imgs_from_videos.py

Extracts images from videos in a specified folder.

**Input:**
- `--num_images`: Number of images to extract from each video (default: 1)
- `--video_dir_path`: Path to the folder containing videos (default: ../background_videos)
- `--save_dir_path`: Path to save extracted images (default: ../extracted_background_images/)

## Usage Instructions

Please follow the input parameters for each script to use them effectively. Make sure to provide the correct paths and files as required. This README serves as a comprehensive guide to understand the functionality of each script and their intended usage within the project.


**Steps for training:**
- Run dataset_analysis_train.py : Get the labels.csv file
- Run extract_images_from_video.py :  Get the background images
- Run create_images_num.py: Get the artificial images
- Create the train.txt file:
    - rm train.txt
    - rm val.txt
    - cp ../ref_images/train.txt train.txt
    - cp ../ref_images/val.txt val.txt
    - ls /datadrive/data/artificial_data/images/| sed "s|^|/datadrive/data/artificial_data/images/|" >> train.txt
- Run the yolov8 training
    - yolo task=detect mode=train model=yolov8s.pt data=/datadrive/data/data_gen_images/data.yaml epochs=400 imgsz=640 batch=40 workers=8 project=../../../datadrive/data/hiba name=yolov8_data_gen_40b_400e_eqclasses_600background
