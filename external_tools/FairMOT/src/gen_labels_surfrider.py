import json 
import os 
import cv2
annotations_path = '/home/infres/chagneux/datasets/surfrider_data/images/annotations'
images_path='/home/infres/chagneux/datasets/surfrider_data/images/images'
data_dir = 'src/data'

for split in ['train', 'val']:
    labels_dir = os.path.join(data_dir,'surfrider','labels_with_ids',split)
    image_target_dir = os.path.join(data_dir,'surfrider','images',split)


    with open(os.path.join(annotations_path,'instances_{}.json'.format(split)),'r') as f:
        annotations = json.load(f)
        
    images = sorted([image for image in annotations['images']],key=lambda x: x['id'])

    with open(os.path.join(data_dir,'surfrider.{}'.format(split)),'w') as f:
        f.writelines(os.path.join('images', split, str(image_nb)+'.'+image['file_name'].split('.')[1]+'\n') for image_nb, image in enumerate(images))

    object_id = 0
    for image_nb,image in enumerate(images):
        image_filename = os.path.join(images_path, image['file_name'])
        img_shape = cv2.imread(image_filename).shape
        img_width = img_shape[1]
        img_height = img_shape[0]
        os.symlink(image_filename, os.path.join(image_target_dir,str(image_nb)+'.'+image['file_name'].split('.')[1]))

        annotations_for_image = [annotation for annotation in annotations['annotations'] if annotation['image_id'] == image['id']]
        with open(os.path.join(labels_dir,str(image_nb)+'.txt'),'w') as f:
            for annotation in annotations_for_image:
                left, top, w, h = annotation['bbox']
                x_center = left + w/2
                y_center = top + h/2
                f.write('0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(object_id, x_center/img_width, y_center/img_height, w/img_width, h/img_height))
                object_id+=1

