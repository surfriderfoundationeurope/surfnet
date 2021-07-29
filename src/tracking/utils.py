from scipy.stats import multivariate_normal
import numpy as np 
import os
from tqdm import tqdm


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

def init_trackers(engine, detections, frame_nb, state_variance, observation_variance, delta):
    trackers = []

    for detection in detections:
        tracker_for_detection = engine(frame_nb, detection, state_variance, observation_variance, delta)
        trackers.append(tracker_for_detection)

    return trackers

def exp_and_normalise(lw):
    w = np.exp(lw - lw.max())
    return w / w.sum()

def in_frame(position, shape, border=0.02):


    shape_x = shape[1]
    shape_y = shape[0]
    x = position[0]
    y = position[1]

    return x > border*shape_x and x < (1-border)*shape_x and y > border*shape_y and y < (1-border)*shape_y
    
def gather_filenames_for_video_in_annotations(video, images, data_dir):
    images_for_video = [image for image in images
                        if image['video_id'] == video['id']]
    images_for_video = sorted(
        images_for_video, key=lambda image: image['frame_id'])

    return [os.path.join(data_dir, image['file_name'])
                 for image in images_for_video]

def get_detections_for_video(reader, detector):

    detections = []
    for frame in tqdm(reader):

        detections_for_frame = detector(frame)
        if len(detections_for_frame): detections.append(detections_for_frame)
        else: detections.append(np.array([]))

    return detections

def resize_external_detections(detections, ratio):

    for detection_nb in range(len(detections)):
        detection = detections[detection_nb]
        if len(detection):
            detection = np.array(detection)[:,:-1]
            detection[:,0] = (detection[:,0] + detection[:,2])/2
            detection[:,1] = (detection[:,1] + detection[:,3])/2
            detections[detection_nb] = detection[:,:2]/ratio
    return detections
        
def write_tracking_results_to_file(results, ratio_x, ratio_y, output_filename):
    output_file = open(output_filename+'.txt', 'w')

    for result in results:
        output_file.write('{},{},{},{},{},{},{},{},{},{}\n'.format(result[0]+1,
                                                                result[1]+1,
                                                                ratio_x * result[2],
                                                                ratio_y * result[3],
                                                                -1,
                                                                -1,
                                                                1,
                                                                -1,
                                                                -1,
                                                                -1))

    output_file.close()

    
class FramesWithInfo:
    def __init__(self, frames, output_shape=None):
        self.frames = frames
        if output_shape is None: 
            self.output_shape = frames[0].shape[:-1][::-1]
        else: self.output_shape = output_shape
        self.end = len(frames)
        self.read_head = 0
    
    def __next__(self):
        if self.read_head < self.end:
            frame = self.frames[self.read_head]
            self.read_head+=1
            return frame

        else: 
            raise StopIteration
    
    def __iter__(self):
        return self

