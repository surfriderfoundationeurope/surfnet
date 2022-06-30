from pycocotools.coco import COCO
import numpy as np
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from PIL import Image, ExifTags
import os 
import cv2

def draw_bbox(image, anns, ratio):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    # """
    for ann in anns:

        [bbox_x, bbox_y, bbox_w, bbox_h] = (ratio*np.array(ann['bbox'])).astype(int)
        
        cv2.rectangle(image, (bbox_x,bbox_y),(bbox_x+bbox_w,bbox_y+bbox_h), color=(255,0,0),thickness=3)

    return image

dir = 'data/images'

ann_dir = os.path.join(dir,'annotations')
data_dir = os.path.join(dir,'images')
ann_file = os.path.join(ann_dir, 'instances_train.json')
coco = COCO(ann_file)

imgIds = np.array(coco.getImgIds())
print('{} images loaded'.format(len(imgIds)))
permutation = np.random.permutation(imgIds.shape[0])

permutation = np.random.permutation(imgIds)
for imgId in permutation:
    image = coco.loadImgs(ids=[imgId])[0]
    try:
        image = Image.open(os.path.join(data_dir,image['file_name']))
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        
        exif = image._getexif()
        if exif is not None:
            if exif[orientation] == 3:
                image=image.rotate(180, expand=True)
            elif exif[orientation] == 6:
                image=image.rotate(270, expand=True)
            elif exif[orientation] == 8:
                image=image.rotate(90, expand=True)

    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass
    image = cv2.cvtColor(np.array(image.convert('RGB')),  cv2.COLOR_RGB2BGR)
    annIds = coco.getAnnIds(imgIds=[imgId])
    anns = coco.loadAnns(ids=annIds)
    h,w = image.shape[:-1]
    target_h = 1080
    ratio = target_h/h
    target_w = int(ratio*w) 
    image = cv2.resize(image,(target_w,target_h))
    image = draw_bbox(image,anns,ratio)
    cv2.imshow('image',image)
    cv2.waitKey(0)




