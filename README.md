# yolov5_train_test
Train yolov5 object detection model 

## Data

#### I. CSV Files

Two CSVs, one for the bounding boxes and one for the images.

- The bounding box file contains the bounding box info. : the label ID, the category number of the object, the ID of the associated image, the x; y; width and height coordinates, as well as the "id_creator_fk","createdon". There are 8098 labels.

- The images file contains information about: the ID info, the view of the picture, the image quality, the context as well as the "id_creator_fk", 	"createdon", "container_url", "blob_name". There are 7 439 images. 

### II. Instances Files

Contains information on the images, the annotations (labels) as well as the categories of the labels. There are three files : whole, train and validation. 

- For the images we have an ID number and a filename for each image
    - ex: {"id": 3526, "file_name": "img1269.jpg"} 

- For the annotations we have information on the ID of the label, the ID of the image, as well as the coordinates of the bounding box 
    - ex: {"id": 6439, "image_id": 3529, "bbox": [969, 2288, 221, 154], "category_id"} 

 - For the categories we have the ID of the category and a supercategory. We note that there are 2 supercategories : unknown and trash 
    - ex: {"id": 1, "name": "Sheet / tarp / plastic bag / fragment", "supercategory": "trash"} 

The COCO function can be used in order to read these instances files.

The coco format is a specific JSON structure that dictates how labels and metadata are saved for an image dataset. It explains how the annotations and the image metadata are stored on the platform. 
### 

## Training 

### 1) Installing the requirements and import various packages needed. 

```
pip install -r requirements.txt
````

We also open the csv files we previously downloaded. 
```
df_bboxes = pd.read_csv("bounding_boxes_202205231416.csv")
df_images = pd.read_csv("images_for_labelling_202205231708.csv")
```
### 2) Data Processing

import the functions :
- coco2yolo
- shapping_bboxes
- image_orientation  
- get_df_train_val 
- path_existance 
- get_date
- get_train_valid

```
df_data = get_df_train_val("instances_val.json", "images2label" , df_images)

df_data = get_date(df_data)

train_files, val_files = get_train_valid(df_data)
```

### 3) 

### 4) Launching YOLOv5

Change the parameters according to your training: 
- batch size ()
- nb. of epochs 
- weights
- workers
- project
- name 
````
yolov5/data/hyps/hyp.scratch.yaml" --batch 32 --epochs 100 --data "data.yaml" --weights "yolov5s.pt" --workers 23 --project "yolo_bene" --name "yolo_ben_100" --exist-ok 
```