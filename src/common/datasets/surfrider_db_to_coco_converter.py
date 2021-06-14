import json
import os
annotation_filename = 'data/images/bounding_boxes.json'

with open(annotation_filename, 'r') as f:
    annotations = json.load(f)['bounding_boxes']
    # annotations = [annotation for annotation in annotations if annotation['createdon'].startswith('2021-06-08')]

image_name_conversion_filename = 'data/images/images_for_labelling.json'

with open(image_name_conversion_filename, 'r') as f: 
    image_name_conversion_table = json.load(f)['images_for_labelling']
    
images_id_refs = list(set([annotation['id_ref_images_for_labelling'] for annotation in annotations]))
image_dbid_to_cocoid = {image_dbid:image_cocoid for image_cocoid, image_dbid in enumerate(images_id_refs)}

image_idref_to_image_filename = {image['id']:image['filename'] for image in image_name_conversion_table}

coco_images = [{'id':image_cocoid, 'file_name':image_idref_to_image_filename[image_dbid]} for image_dbid, image_cocoid in image_dbid_to_cocoid.items()]

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

with open('annotations_from_db.json','w') as f:
    json.dump(coco, f)



