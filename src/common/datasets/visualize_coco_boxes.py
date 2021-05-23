from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from PIL import Image, ImageFile
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
        print(ann['bbox'])
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4,2))
        polygons.append(Polygon(np_poly))
        color.append(c)

    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
    ax.add_collection(p)

dir = 'data/images'

ann_dir = os.path.join(dir,'annotations')
data_dir = os.path.join(dir,'Images_md5')
ann_file = os.path.join(ann_dir, 'instances_train.json')
coco = COCO(ann_file)

imgIds = np.array(coco.getImgIds())
# permutation = np.random.permutation(imgIds.shape[0])

plt.ion()

for imgId in imgIds:
    image = coco.loadImgs(ids=[imgId])[0]
    plt.imshow(Image.open(os.path.join(data_dir,image['file_name'])))
    
    annIds = coco.getAnnIds(imgIds=[imgId],catIds=[1])
    anns = coco.loadAnns(ids=annIds)
    draw_bbox(anns)
    plt.show()
    while not plt.waitforbuttonpress(): continue
    plt.cla()




