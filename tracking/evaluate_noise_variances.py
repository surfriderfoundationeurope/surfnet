import os
import json 
import numpy as np
from tqdm.std import tqdm
from tracking import read_and_resize, compute_flow
from tqdm import tqdm
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def compute_translation_variances(annotation_file, data_dir, downsampling_factor):
    with open(annotation_file,'r') as f:
        annotations = json.load(f)

    errors = []
    for video in tqdm(annotations['videos'],desc='Video'):
        images_for_video = [image for image in annotations['images'] if image['video_id'] == video['id']]
        images_for_video = sorted(images_for_video, key=lambda image: image['frame_id'])

        filenames = [os.path.join(data_dir, image['file_name'])
                        for image in images_for_video]


        for frame_nb in range(len(filenames)-1):
            frame0, old_shape, new_shape = read_and_resize(filenames[frame_nb])
            frame1, _, _ = read_and_resize(filenames[frame_nb+1])
            flow01 = compute_flow(frame0, frame1, downsampling_factor=downsampling_factor)
            image0 = images_for_video[frame_nb]
            image1 = images_for_video[frame_nb+1]
            ratio_y = (new_shape[0] // downsampling_factor) / old_shape[0] 
            ratio_x = (new_shape[1] // downsampling_factor) / old_shape[1]
            annotations0 = [annotation for annotation in annotations['annotations'] if annotation['image_id'] == image0['id']]
            annotations1 = [annotation for annotation in annotations['annotations'] if annotation['image_id'] == image1['id']]
            for annotation0 in annotations0:
                for annotation1 in annotations1:
                    if annotation0['track_id'] == annotation1['track_id']:
                        left, top, w, h = annotation0['bbox']
                        center0 = np.array([ratio_x*(left+w/2), ratio_y*(top+h/2)])
                        moved_center0 = center0 + flow01[int(center0[1]),int(center0[1])]

                        left, top, w, h = annotation1['bbox']
                        center1 = np.array([ratio_x*(left+w/2), ratio_y*(top+h/2)])

                        errors.append(center1-moved_center0)


    np.save('translation_errors', np.array(errors))


def get_external_detections_for_video(video, nb_frames, external_detections_dir):
        detections = []

        detections_filename = os.path.join(external_detections_dir, video['file_name']+'.txt')
        with open(detections_filename,'r') as f:
            detections_read = [detection.split(',') for detection in f.readlines()]
        detections_from_file = defaultdict(list)
        for detection in detections_read: 
            detections_from_file[int(detection[0])].append([float(detection[2]),float(detection[3])])

        detections_from_file = {k:np.array(v) for k,v in detections_from_file.items()}

        for frame_nb in range(nb_frames):
            if frame_nb+1 in detections_from_file.keys():
                detections.append(detections_from_file[frame_nb+1])
            else:
                detections.append(np.array([]))
        return detections


def compute_observation_variances(annotation_file, data_dir, downsampling_factor, external_detections_dir):
    with open(annotation_file,'r') as f:
        annotations = json.load(f)

    errors = []
    for video in tqdm(annotations['videos'],desc='Video'):
        # video = [video_annotation for video_annotation in annotations['videos'] if video_annotation['file_name'] == 'leloing__2'][0] # debug

        images_for_video = [image for image in annotations['images'] if image['video_id'] == video['id']]
        images_for_video = sorted(images_for_video, key=lambda image: image['frame_id'])

        filenames = [os.path.join(data_dir, image['file_name'])
                        for image in images_for_video]
        _, old_shape, new_shape = read_and_resize(filenames[0])
        ratio_y = (new_shape[0] // downsampling_factor) / old_shape[0] 
        ratio_x = (new_shape[1] // downsampling_factor) / old_shape[1]
        nb_frames = len(filenames)
        detections = get_external_detections_for_video(video, nb_frames, external_detections_dir)

        ratio_y = (new_shape[0] // downsampling_factor) / old_shape[0]
        ratio_x = (new_shape[1] // downsampling_factor) / old_shape[1] 

        detections_resized = []
        for detection in detections: 
            if len(detection):
                detections_resized.append(np.array([ratio_x,ratio_y])*detection)
            else:
                detections_resized.append(detection)
        detections = detections_resized

        for frame_nb in range(nb_frames):
            bboxes_for_frame = [annotation['bbox'] for annotation in annotations['annotations'] if annotation['image_id'] == images_for_video[frame_nb]['id']]
            gt_dets_frame = np.array([(ratio_x *(bbox[0]+bbox[2]/2), ratio_y*(bbox[1]+bbox[3]/2)) for bbox in bboxes_for_frame])
            dets_frame = detections[frame_nb]
            if len(gt_dets_frame) and len(dets_frame): 
                distance_matrix = cdist(gt_dets_frame, dets_frame, metric='sqeuclidean')
                row_inds, col_inds = linear_sum_assignment(distance_matrix)
                for row_ind, col_ind in zip(row_inds,col_inds):
                    errors.append(distance_matrix[row_ind, col_ind])

    np.save('observation_errors', np.array(errors))





annotation_file = 'data/new_synthetic_videos_dataset/annotations/annotations_val.json'
data_dir = 'data/new_synthetic_videos_dataset/data'
downsampling_factor = 4
external_detections_dir = 'data/detector_results/surfrider-test-longer/CenterTrack'
# compute_translation_variances(annotation_file,data_dir, downsampling_factor)

# translation_errors = np.load('translation_errors.npy')
# var_x = sum(translation_error[0]**2 for translation_error in translation_errors)/len(translation_errors)
# var_y = sum(translation_error[1]**2 for translation_error in translation_errors)/len(translation_errors)

# np.save('state_variance',np.array([var_x,var_y]))

# compute_observation_variances(annotation_file, data_dir, downsampling_factor=4, external_detections_dir=external_detections_dir)

observation_errors = np.load('observation_errors.npy')

var_xy = np.median(observation_errors)
np.save('observation_variance',np.array([var_xy,var_xy]))

