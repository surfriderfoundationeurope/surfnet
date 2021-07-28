import os 
import json
annotation_dir = '/home/infres/chagneux/repos/surfnet/data/new_synthetic_videos_dataset/annotations'
from collections import defaultdict

results_dir = os.path.join(annotation_dir, 'kitti_versions')
annotations_val = os.path.join(annotation_dir, 'annotations_val.json')

os.mkdir(results_dir)

with open(annotations_val,'r') as f:
    annotations = json.load(f)

# video_to_images = defaultdict(list)
# for image in annotations['images']:
#     video_to_images[image['video_id']].append(image)

alpha = -10
loc = [-1000, -1000, -1000]
rot_y = -10
score = -1
seqmap_file = open(os.path.join(results_dir,'evaluate_tracking.seqmap.training'),'w')

cat_ids_to_cat_names = {1:'Car'}
for video in annotations['videos']:
    f = open(os.path.join(results_dir,video['file_name']+'.txt'),'w')
    images_for_vid = sorted([image for image in annotations['images'] if image['video_id']==video['id']],key = lambda image: image['frame_id'])
    for i,image in enumerate(images_for_vid):
        annotations_for_image = [annotation for annotation in annotations['annotations'] if annotation['image_id']==image['id']]
        for annotation in annotations_for_image:
            f.write('{} {} {} -1 -1'.format(image['frame_id'] - 1, annotation['track_id'], cat_ids_to_cat_names[annotation['category_id']]))
            f.write(' {:.6f}'.format(alpha))
            f.write(' {:.2f} {:.2f} {:.2f} {:.2f}'.format(
                annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][0]+annotation['bbox'][2], annotation['bbox'][1]+annotation['bbox'][3]))
            f.write(' {:.6f} {:.6f} {:.6f}'.format(
                int(loc[0]), int(loc[1]), int(loc[2])))
            f.write(' {:6f} {:.6f} {:.6f} {:.6f}\n'.format(int(rot_y), score,score,score))
            # else:
            #     f.write(' {:6f} {:.6f} {:.6f} {:.6f}'.format(int(rot_y), score,score,score))

    seqmap_file.write('{} empty 000000 {:06d}\n'.format(video['file_name'], image['frame_id']))
    f.close()




   