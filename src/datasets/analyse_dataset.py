import json 
from pycocotools.coco import COCO
import matplotlib.pyplot as plt


coco = COCO(annotation_file = 'data/images/annotations/instances_multiclass.json')

coco_categories = coco.dataset['categories'][1:]

nb_anns_per_cat = {cat['name']: len(coco.getAnnIds(catIds=[cat['id']])) for cat in coco_categories}
nb_anns_per_cat = {k:v for k,v in sorted(nb_anns_per_cat.items(), key=lambda x: x[1], reverse=True)}
cat_names = list(nb_anns_per_cat.keys())
nb_images = list(nb_anns_per_cat.values())

plt.bar(x = cat_names, height = nb_images)
plt.xticks(range(len(cat_names)), cat_names, rotation='vertical')
plt.ylabel('Number of annotations')
plt.tight_layout()
plt.autoscale(True)
plt.savefig('dataset_analysis')