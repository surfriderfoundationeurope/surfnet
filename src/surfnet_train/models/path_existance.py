from datetime import datetime
#from src.surfnet_train.data.data_processing import shaping_bboxes
#from Users.rispal.ZoRaFa.yolov5_train_test.src.surfnet_train.data import image_orientation
import os
import numpy as np
import pandas as pd
import cv2
import datetime
from PIL import Image

def path_existance(img_ids, data_dir, coco, df_images) :

    """ Function which 

    Args:
        img_ids (array): Array with the images IDs created by the coco function. 
        data_dir (file): File with the images. Default to images2labels.
        coco (): Annotation file transformed and prepares data structures 
        df_images (): pd.read_csv("images_for_labelling_202201241120.csv"))

    Returns:
        my_df (data frame):  
    """

    old_filenames  = [] 
    dates          = []
    views          = []
    images_quality = []
    contexts       = []
    all_bboxes     = []
    all_images     = []
    new_filenames  = []
    new_labelnames = []

    for img_id in img_ids:
        image_infos = coco.loadImgs(ids=[img_id])[0]
        
        if os.path.exists(os.path.join(data_dir, image_infos['file_name'])):
            
            # concatenate the data directory path and the files from the coco transformation ; if it exists, we compute the if loop.

            date_creation  = df_images.loc[df_images["filename"] == image_infos["file_name"]]["createdon"].values[0] 
            view           = df_images.loc[df_images["filename"] == image_infos["file_name"]]["view"].values[0]
            image_quality  = df_images.loc[df_images["filename"] == image_infos["file_name"]]["image_quality"].values[0]
            context        = df_images.loc[df_images["filename"] == image_infos["file_name"]]["context"].values[0]

            date_time_obj = datetime.datetime.strptime(date_creation, '%Y-%m-%d %H:%M:%S')

            old_filenames.append(image_infos["file_name"])
            dates.append(date_time_obj)
            views.append(view)
            images_quality.append(image_quality)
            contexts.append(context)

            # in the loop we put the info of the image corresponding to the date, the type of view, 
            # the quality and the context to the empty lists created in previous function.

            image = Image.open(os.path.join(data_dir,image_infos['file_name'])) 
            # image is opened and identified it returns an image object 

            image_orientation(image) 
            # calls function which gives the images that have a specified orientation the same orientation

            image    = np.array(image) #cv2.cvtColor(np.array(image.convert('RGB')),  cv2.COLOR_RGB2BGR)
            ann_ids  = coco.getAnnIds(imgIds=[img_id]) # gets us the ids of the annotations
            anns     = coco.loadAnns(ids=ann_ids) 
            # list with : ID of the label, ID of the image, bounding box coordinates and the category ID
            h, w     = image.shape[:-1] 
            target_h = 1080 # the target height of the image 
            ratio    = target_h/h # We get the ratio of the target and the actual height 
            target_w = int(ratio*w) 
            image    = cv2.resize(image,(target_w,target_h)) # we resize the image with the new target shapes 
            h, w     = image.shape[:-1]  
            yolo_annot = []

            shaping_bboxes(anns, ratio, target_h, target_w)
            # calls function which shapes the bounding boxes as normalized
            
            basename  = os.path.splitext(image_infos['file_name'])[0]
            file_name = str(image_infos['id']) + "-" + basename

            img_file_name   = os.path.join("./images", file_name) + ".jpg"
            label_file_name = os.path.join("./labels", file_name) + ".txt"
            
            # Save Label
            with open(label_file_name, 'w') as f:
                f.write('\n'.join(yolo_annot))

            img_to_save = Image.fromarray(image)
            # Save image
            img_to_save.save(img_file_name)

            all_bboxes.append(yolo_annot)
            new_filenames.append(img_file_name)
            new_labelnames.append(label_file_name)
            all_images.append(img_to_save)
            
    my_list = list(zip(old_filenames, dates, views, images_quality, contexts, new_filenames, new_labelnames, all_images, all_bboxes))
    my_df   = pd.DataFrame(my_list, columns=['old_path', 'date', 'view', 'quality','context', 'img_name', 'label_name', 'img', 'bboxes'])

    return(my_df)