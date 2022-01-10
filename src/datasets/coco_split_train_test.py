import json 
import numpy as np 
import os 


def subset_from_image_ids(coco_dataset, image_ids):
    new_annotations = list() 
    new_images = list()
    annotation_nb = 0 
    for image_nb, image_id in enumerate(image_ids):
        image = next(image.copy() for image in coco_dataset['images'] if image['id'] == image_id).copy()
        image['id'] = image_nb 
        new_images.append(image)
        annotations_for_image = [annotation.copy() for annotation in coco_dataset['annotations'] if annotation['image_id'] == image_id]
        for annotation in annotations_for_image: 
            annotation['image_id'] = image_nb
            annotation['id'] = annotation_nb
            new_annotations.append(annotation)
            annotation_nb+=1
    categories = coco_dataset['categories']
    return {'images':new_images,'annotations':new_annotations,'categories':categories}

train_proportion = 0.9
annotations_dir = 'data/images/annotations'
with open(os.path.join(annotations_dir,'instances.json'),'r') as f:
    full_dataset = json.load(f)

image_ids = [image['id'] for image in full_dataset['images']]
image_ids = np.array(image_ids)[np.random.permutation(len(image_ids))].tolist()
image_ids_train = image_ids[:int(train_proportion*len(image_ids))]
image_ids_val = image_ids[int(train_proportion*len(image_ids)):]

with open(os.path.join(annotations_dir,'instances_train.json'),'w') as f: 
    json.dump(subset_from_image_ids(full_dataset, image_ids_train),f)

with open(os.path.join(annotations_dir,'instances_val.json'),'w') as f: 
    json.dump(subset_from_image_ids(full_dataset, image_ids_val),f)