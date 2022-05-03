def coco2yolo(bbox, image_height=1080, image_width=1080):
    """
    coco  => [x1, y1, w, h]
    yolo  => [xmid, ymid, w, h] (normalized)
    """
    
    bbox = bbox.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int

    bbox[[0, 2]] = bbox[[0, 2]]/ image_width
    bbox[[1, 3]] = bbox[[1, 3]]/ image_height
    
    bbox[[0, 1]] = bbox[[0, 1]] + bbox[[2, 3]]/2
    
    return bbox

# Function to normalize the representation of the bounding box, 
# such that there are in the yolo format 