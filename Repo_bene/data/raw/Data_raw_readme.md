I. CSV

Two csvs, one for the bounding boxes and one for the images.

- The bounding box file contains : "id","id_creator_fk","createdon","id_ref_trash_type_fk","id_ref_images_for_labelling","location_x","location_y","width","height" : the information in the csv for the bounding boxes. There are 8098 labels.

- The images file contains information about: the ID info, the view of the picture, the image quality, the context. There are 5371 images. 

II. Instances File 

Contains information on the images, the annotations (labels) as well as the categories of the labels. 

- For the images we have an ID number and a filename for each image
    # ex: {"id": 3526, "file_name": "img1269.jpg"} 

- For the anotations we have information on the ID of the label, the ID of the image, as well as the coordinates of the bounding box 
    # ex: {"id": 6439, "image_id": 3529, "bbox": [969, 2288, 221, 154], "category_id"} 

 - For the categories we have the ID of the category and a supercategory. We note that there are 2 supercategories : unknown and trash 
    # ex: {"id": 1, "name": "Sheet / tarp / plastic bag / fragment", "supercategory": "trash"} 

The COCO function can be used in order to read these instances files.

The coco format is a specific JSON structure that dictates how labels and metadata are saved for an image dataset. It explains how the annotations and the image metadata are stored on the platform. 