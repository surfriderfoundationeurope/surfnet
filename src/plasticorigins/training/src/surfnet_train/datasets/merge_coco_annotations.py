import json 

annotation_filename_1 = 'data/images/annotations/instances.json'
annotation_filename_2 = 'data/images/annotations/instances_remaining.json'

with open(annotation_filename_1,'r') as f: 
    annotations_1 = json.load(f)
with open(annotation_filename_2,'r') as f: 
    annotations_2 = json.load(f)



new_images = list()
image_ids_1_to_image_ids = dict()
image_nb = 0
for image in annotations_1['images']:
    image_ids_1_to_image_ids[image['id']] = image_nb
    image['id'] = image_nb
    new_images.append(image)
    image_nb+=1


image_ids_2_to_image_ids = dict()
for image in annotations_2['images']:
    image_ids_2_to_image_ids[image['id']] = image_nb
    image['id'] = image_nb
    new_images.append(image)
    image_nb+=1


new_annotations = []
annotation_nb = 0
for annotation in annotations_1['annotations']:
    annotation['image_id'] = image_ids_1_to_image_ids[annotation['image_id']]
    annotation['id'] = annotation_nb
    new_annotations.append(annotation)
    annotation_nb+=1

for annotation in annotations_2['annotations']:
    annotation['image_id'] = image_ids_2_to_image_ids[annotation['image_id']]
    annotation['id'] = annotation_nb
    new_annotations.append(annotation)
    annotation_nb+=1


annotations = {'annotations':new_annotations,'images':new_images,'categories':annotations_1['categories']}

with open('data/images/annotations/instances.json','w') as f:

    json.dump(annotations, f)