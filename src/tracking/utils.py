from common.utils import load_my_model
from extension.models import SurfNet
from scipy.stats import multivariate_normal
import numpy as np 
import cv2
import os
from tqdm import tqdm
import torch
from common.utils import transform_test_CenterNet, nms
import pickle
from collections import defaultdict


class GaussianMixture(object):
    def __init__(self, means, covariance, weights):
        self.components = [multivariate_normal(
            mean=mean, cov=covariance) for mean in means]
        self.weights = weights

    def pdf(self, x):
        result = 0
        for weight, component in zip(self.weights, self.components):
            result += weight*component.pdf(x)
        return result

    def logpdf(self, x):
        return np.log(self.pdf(x))

    def cdf(self, x):
        result = 0
        for weight, component in zip(self.weights, self.components):
            result += weight*component.cdf(x)
        return result


def init_trackers(engine, detections, frame_nb, state_variance, observation_variance, stop_tracking_threshold):
    trackers = []

    for detection in detections:
        tracker_for_detection = engine(frame_nb, detection, state_variance, observation_variance, stop_tracking_threshold=stop_tracking_threshold)
        trackers.append(tracker_for_detection)

    return trackers

def exp_and_normalise(lw):
    w = np.exp(lw - lw.max())
    return w / w.sum()

def in_frame(position, shape):

    shape_x = shape[1]
    shape_y = shape[0]
    x = position[0]
    y = position[1]

    return x > 0 and x < shape_x-1 and y > 0 and y < shape_y-1
    
def gather_filenames_for_video_in_annotations(video, images, data_dir):
    images_for_video = [image for image in images
                        if image['video_id'] == video['id']]
    images_for_video = sorted(
        images_for_video, key=lambda image: image['frame_id'])

    return [os.path.join(data_dir, image['file_name'])
                 for image in images_for_video]

def detect_base_extension(frame, threshold, base_model, extension_model):

    frame = transform_test_CenterNet()(frame).to('cuda').unsqueeze(0)
    base_result = base_model(frame)[-1]['hm']
    extension_result = torch.sigmoid(extension_model(base_result))
    detections = nms(extension_result).gt(threshold).squeeze()

    return torch.nonzero(detections).cpu().numpy()[:, ::-1]

def detect_base(frame, threshold, base_model):

    frame = transform_test_CenterNet()(frame).to('cuda').unsqueeze(0)
    base_result = torch.sigmoid(base_model(frame)[-1]['hm'])
    detections = nms(base_result).gt(threshold).squeeze()

    return torch.nonzero(detections).cpu().numpy()[:, ::-1], base_result.squeeze().cpu().numpy()

def detect_internal(reader, detector):

    detections = []
    heatmaps = []
    for frame in tqdm(reader):

        detections_for_frame, heatmap_for_frame = detector(frame)
        heatmaps.append(heatmap_for_frame)
        if len(detections_for_frame): detections.append(detections_for_frame)
        else: detections.append(np.array([]))

    reader.init()
    return detections, heatmaps

def detect_external(detections_filename, heatmaps_filename, file_type='mot', nb_frames=None):

    if file_type == 'mot':
        with open(detections_filename, 'r') as f:
            detections_read = [detection.split(',') for detection in f.readlines()]
        detections_from_file = defaultdict(list)
        for detection in detections_read:
            detections_from_file[int(detection[0])].append(
                [float(detection[2]), float(detection[3])])

        detections_from_file = {k: np.array(v)
                                for k, v in detections_from_file.items()}

        detections = []

        for frame_nb in range(nb_frames):
            if frame_nb+1 in detections_from_file.keys():
                detections.append(detections_from_file[frame_nb+1])
            else:
                detections.append(np.array([]))


    elif file_type == 'CenterTrackpickle':
        with open(detections_filename, 'rb') as f:
            detections_read = pickle.load(f)
        
        detections = []

        for detections_for_frame in detections_read.values():
            if len(detections_for_frame): 
                detections.append(np.concatenate([detection['ct'].reshape(1,2) for detection in detections_for_frame]))
            else: detections.append(np.array([]))
    
    elif file_type == 'simplepickle':
        with open(detections_filename, 'rb') as f:
            detections = pickle.load(f)
        with open(heatmaps_filename,'rb') as f:
            heatmaps = pickle.load(f)


    return detections, heatmaps