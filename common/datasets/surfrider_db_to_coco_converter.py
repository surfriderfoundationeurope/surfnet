import json
import os
annotation_filename = 'data/surfrider_images/annotations/bounding_boxes_202105170929.json'

with open(annotation_filename, 'r') as f:
    annotations = json.load(f)['bounding_boxes']

images_filenames = list(set([annotation['id_ref_images_for_labelling'] for annotation in annotations]))
image_dbid_to_cocoid = {image_dbid:image_cocoid for image_cocoid, image_dbid in enumerate(images_filenames)}


coco_images = [{'id':image_cocoid, 'file_name':image_dbid+'.jpg'} for image_dbid, image_cocoid in image_dbid_to_cocoid.items()]

coco_categories = [{'id':0,'name':'__background__','supercategory':'unknown'},
                   {'id':1,'name':'trash','supercategory':'unknown'}]

coco_annotations = list()

for annotation_id, annotation in enumerate(annotations):
    image_dbid = annotation['id_ref_images_for_labelling']
    bbox = [annotation['location_x'], annotation['location_y'], annotation['width'], annotation['height']]

    coco_annotations.append({'id':annotation_id,
                             'image_id':image_dbid_to_cocoid[image_dbid],
                             'bbox':bbox,
                             'category_id':1})

coco = {'images':coco_images,'annotations':coco_annotations,'categories':coco_categories}

with open('annotations.json','w') as f:
    json.dump(coco, f)



