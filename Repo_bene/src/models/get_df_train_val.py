from Repo_bene.src.models.path_existance import path_existance

def get_df_train_val(annotation_file):

    """Transforms the labels and images to the same format in order to be used by the Yolov5 algorithm. 
  
    Args: 
        annotation_file (json instances file): Annotation file which contains information on the images, 
        the annotations (labels) and categories of the labels. 

    Returns:
        my_df (data frame): Data frame with columns : old_path, date, view, quality, context, img_name,
         label_name, image and bounding box. 
    """
    
    coco = COCO(annotation_file) # transform the file using a coco function where the COCO function 
                                 # loads a coco annotation file and prepares data structures 
    # gives the annotations into a coco api form ; helps the user in extracting annotations conveniently

    old_filenames  = [] 
    dates          = []
    views          = []
    images_quality = []
    contexts       = []
    all_bboxes     = []
    all_images     = []
    new_filenames  = []
    new_labelnames = []

    img_ids = np.array(coco.getImgIds()) # creates an array with the images IDs processed by coco

    path_existance(img_ids) # calls function

    return (my_df)