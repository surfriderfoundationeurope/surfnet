from Repo_bene.src.models.coco2yolo import coco2yolo

def shaping_bboxes(anns:list):
    """Function in charge of shaping the bounding boxes, normalized via the coco2yolo function.

    Args:
        anns (list): List with ID of the label, ID of the image, bounding box coordinates and the category ID
    
    Returns: yolo_annot a list with the coordinates of the bboxes and their associated label. 
        _type_: list
    """
    yolo_annot = []

    for ann in anns:
        cat = ann['category_id'] - 1 # gets the actual categories, according to the initial yaml 
        [bbox_x, bbox_y, bbox_w, bbox_h] = (ratio*np.array(ann['bbox'])).astype(int) 
        # gets the bboxes coordinates * the ratio = (1080 /height of the image)
        bbox = np.array([bbox_x, bbox_y, bbox_w, bbox_h]) # creates array with the bboxes coordinates 
        yolo_bbox = coco2yolo(bbox, target_h, target_w) # calls coco2yolo function to normalize the coordinates 
        yolo_str  = str(cat) + " " + " ".join(yolo_bbox.astype(str)) 
        # gives the category of thee label and the coordinates of the bboxes 
        yolo_annot.append(yolo_str)

    return(yolo_annot)
    # list with the coordinates of the bboxes and their associated label

