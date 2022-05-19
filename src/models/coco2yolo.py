def coco2yolo(bbox:list, image_height:int=1080, image_width:int=1080):
    
    """Function to normalize the representation of the bounding box, such that there are in the yolo format (normalized)

    Args:
        bbox (list): Coordinates of the bounding box : x, y, w and h coordiates.
        image_height (int, optional): Height of the image. Defaults to 1080.
        image_width (int, optional): Width of the image. Defaults to 1080.

    Returns: Normalized bounding box coordinates : values between 0-1
        _type_: array 
    """
    
    bbox = bbox.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int

    bbox[[0, 2]] = bbox[[0, 2]]/ image_width
    bbox[[1, 3]] = bbox[[1, 3]]/ image_height
    
    bbox[[0, 1]] = bbox[[0, 1]] + bbox[[2, 3]]/2
    
    return bbox