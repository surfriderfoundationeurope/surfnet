from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image, ExifTags
import os 

def draw_bbox(anns):
    """
    Display the specified annotations.
    :param anns (array of object): annotations to display
    :return: None
    """
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4,2))
        polygons.append(Polygon(np_poly))
        color.append(c)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)

dir = 'data/images/new'

ann_dir = os.path.join(dir,'annotations')
data_dir = os.path.join(dir,'Images_md5')
ann_file = os.path.join(ann_dir, 'instances_train.json')
coco = COCO(ann_file)

imgIds = np.array(coco.getImgIds())
# permutation = np.random.permutation(imgIds.shape[0])

plt.ion()

for imgId in imgIds:
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
    plt.imshow(image)
    annIds = coco.getAnnIds(imgIds=[imgId])
    anns = coco.loadAnns(ids=annIds)
    draw_bbox(anns)
    plt.show()
    while not plt.waitforbuttonpress(): continue
    plt.cla()
    image.close()




