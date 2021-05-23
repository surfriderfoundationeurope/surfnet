import os 
import json
annotation_dir = '/home/infres/chagneux/repos/surfnet/data/new_synthetic_videos_dataset/annotations'
from collections import defaultdict

results_dir = os.path.join(annotation_dir, 'mot_versions')
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

cat_ids_to_cat_names = {1:'Pedestrian'}

seqmaps = open(os.path.join(results_dir,'surfrider-test.txt'),'w')
seqmaps.write('name\n')

for video in annotations['videos']:
    dir_for_video = os.path.join(results_dir, video['file_name'])
    os.mkdir(dir_for_video)
    seqmap_file = open(os.path.join(dir_for_video,'seqinfo.ini'),'w')

    gt_dir_for_video = os.path.join(dir_for_video,'gt')
    os.mkdir(gt_dir_for_video)
    f = open(os.path.join(gt_dir_for_video, 'gt.txt'),'w')

    images_for_vid = sorted([image for image in annotations['images'] if image['video_id']==video['id']],key = lambda image: image['frame_id'])
    
    for i,image in enumerate(images_for_vid):
        annotations_for_image = [annotation for annotation in annotations['annotations'] if annotation['image_id']==image['id']]
        for annotation in annotations_for_image:
            left, top, w, h = annotation['bbox']
            center_x, center_y = left+w/2, top+h/2
            f.write('{},{},{},{},{},{},{},{}\n'.format(image['frame_id'],
                                                            annotation['track_id']+1, 
                                                            center_x, 
                                                            center_y,
                                                            -1,
                                                            -1,
                                                             1,
                                                            -1,
                                                            -1,
                                                            -1))

    f.close()
    seqmap_file.write('[Sequence]\nname={}\nimDir=img1\nframeRate=30\nseqLength={}\nimWidth=1920\nimHeight=1080\nimExt=.png'.format(video['file_name'],image['frame_id']))
    seqmaps.write('{}\n'.format(video['file_name']))
    seqmap_file.close()

seqmaps.close()




   