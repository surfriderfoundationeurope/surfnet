import json 

with open('data/images/annotations/instances_train.json','r') as f:
    coco_dataset = json.load(f)


coco_dataset['categories'] = [{'id': 2, 'name': '__background__', 'supercategory': 'unknown'}, {'id': 1, 'name': 'trash', 'supercategory': 'unknown'}]

new_annotations = list()

for annotation in coco_dataset['annotations']:
    annotation['category_id'] = 1
    new_annotations.append(annotation)

coco_dataset['annotations'] = new_annotations

with open('data/images/annotations/instances_train_detectron2.json','w') as f:
    coco_dataset = json.load(f)
