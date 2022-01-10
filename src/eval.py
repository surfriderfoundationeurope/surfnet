# from numba.np.ufunc import parallel
# from numba.np.ufunc.decorators import vectorize
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from detection.detect import nms
from tqdm import tqdm
import math
from scipy.spatial.distance import cdist
from tools.misc import load_model, _calculate_euclidean_similarity
from detection.coco_utils import get_surfrider
from detection.transforms import TrainTransforms, ValTransforms


def prec_recall_for_thres(thres, thres_nb, gt, pred, radius):

    detections = (pred >= thres)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    distances_true_positives = []
    distances_false_positives = []


    for gt_frame, detection_frame in zip(gt, detections):
        true_positives_frame = 0
        false_positives_frame = 0
        false_negatives_frame = 0

        position_positives = np.argwhere(gt_frame)
        position_detections = np.argwhere(detection_frame)
        max_allowed_cost = math.sqrt(detection_frame.shape[0]**2 + detection_frame.shape[1]**2)


        if len(position_positives):
            distance_matrix = cdist(position_positives, position_detections, metric='euclidean')
            assigned_positives_for_detections = np.argmin(distance_matrix, axis=0)
            for positive in range(len(position_positives)):
                assigned_detections = np.argwhere(assigned_positives_for_detections == positive)
                if len(assigned_detections):
                    distances_to_detections = distance_matrix[positive, assigned_detections.squeeze()]

                    if np.isscalar(distances_to_detections):
                        distances_to_detections = np.array([distances_to_detections])

                    similarities = _calculate_euclidean_similarity(distances_to_detections, zero_distance=max_allowed_cost)
                    closest_detection = np.argmax(similarities)
                    if similarities[closest_detection] > 1-radius:
                        true_positives_frame+=1
                        false_positives_frame+=len(assigned_detections)-1
                        distances_true_positives.append(similarities[closest_detection])
                        similarities = np.delete(similarities, closest_detection)
                    else:
                        false_positives_frame+=len(assigned_detections)
                    distances_false_positives.extend(similarities)
                else:
                    false_negatives_frame+=1
        else:
            false_positives_frame+=len(position_detections)

        true_positives += true_positives_frame
        false_positives += false_positives_frame
        false_negatives += false_negatives_frame

    eps = np.finfo(float).eps
    precision = true_positives/(true_positives+false_positives+eps) + eps
    recall = true_positives/(true_positives+false_negatives+eps) + eps

    return thres_nb, precision, recall, distances_true_positives, distances_false_positives

def prec_recall_with_hungarian(gt, pred, thresholds, radius=0.01):
    _ , precision_list, recall_list, distances_true_positives_list, distances_false_positives_list = [], [], [], [], []
    for thres_nb, thres in enumerate(tqdm(thresholds)):
        _ , precision, recall, distances_true_positives, distances_false_positives  = prec_recall_for_thres(thres, thres_nb, gt, pred, radius)
        precision_list.append(precision)
        recall_list.append(recall)
        distances_true_positives_list.append(distances_true_positives)
        distances_false_positives_list.append(distances_false_positives)


    return precision_list, recall_list, distances_true_positives_list, distances_false_positives_list

def plot_pr_curve(precision_list, recall_list, f1, distances_true_positives_list_best_position, distances_false_positives_list_best_position, thresholds, best_position):

    fig, (ax1,ax2) = plt.subplots(1,2)


    color = 'tab:red'
    ax1.set_xlabel('Recall / Threshold')
    ax1.set_ylabel('Precision / F-score')
    ax1.scatter(recall_list, precision_list, color=color, label='PR-curve')


    color = 'tab:blue'
    ax1.scatter(thresholds, f1, color=color, label='F-score')
    ax1.legend(loc = 'upper right')
    ax1.set_title('Detection performance')


    bins=np.linspace(0,1,num=100)
    best_thres = thresholds[best_position]
    best_f1 = f1[best_position]
    best_recall = recall_list[best_position]
    best_precision = precision_list[best_position]

    ax2.hist([distances_true_positives_list_best_position, distances_false_positives_list_best_position], bins=bins, histtype='barstacked')
    ax2.set_title('Distribution of distances')
    ax2.set_xlabel('Similarity to closest ground truth object')
    ax2.set_ylabel('Quantity')



    plt.suptitle('F-score {:.2f} (Precision {:.2f}, Recall {:.2f}) at threshold {:.2f}'.format(best_f1, best_precision, best_recall, best_thres))
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped

    return (fig, (ax1,ax2))

def pr_curve_from_file(filename, show=True):
    plt.close()
    with open(filename, 'rb') as f:
        fig, axes = plot_pr_curve(*pickle.load(f))
        fig.tight_layout()
    if not show:
        with open(filename.strip('.pickle')+'_axes.pickle','wb') as f:
            data = (fig, axes)
            pickle.dump(data,f)
    else:
        plt.show()
    plt.close()

def compute_precision_recall_hungarian(gt, predictions, output_filename='evaluation', enable_nms=False, plot=False):

    if enable_nms:
        predictions = nms(predictions)
    gt = gt.numpy()
    predictions = predictions.numpy()

    thresholds = np.linspace(0,1,100)
    precision_list, recall_list, distances_true_positives_list, distances_false_positives_list = prec_recall_with_hungarian(gt, predictions, thresholds, radius=3)

    precision_list, recall_list = np.array(precision_list), np.array(recall_list)
    f1 = 2*(precision_list*recall_list)/(precision_list+recall_list)
    best_position = np.argmax(f1)

    distances_true_positives_list_best_position = distances_true_positives_list[best_position]
    distances_false_positives_list_best_position = distances_false_positives_list[best_position]

    with open(output_filename+'.pickle','wb') as f:

        data = (precision_list, recall_list, f1, distances_true_positives_list_best_position, distances_false_positives_list_best_position, thresholds, best_position)
        pickle.dump(data,f)

    if plot:
        plot_pr_curve(precision_list, recall_list, f1, distances_true_positives_list_best_position, distances_false_positives_list_best_position, thresholds, best_position)

def compute_model_outputs(arch):

    device = torch.device('cuda')

    transforms = ValTransforms(540, (544,960), 1, 4)
    dataset = get_surfrider('data/images','val',transforms=transforms)
    loader = DataLoader(dataset, shuffle=False, batch_size=16)

    if arch == 'res_18':
        weights_path = 'models/res18_pretrained.pth'
    elif arch == 'mobilenetv3small':
        weights_path = 'models/mobilenet_v3_pretrained.pth'
    elif arch == 'dla_34':
        weights_path = 'models/dla_34_pretrained.pth'

    model = load_model(arch, weights_path, device=device)
    model.eval()

    all_heatmaps = []
    all_gt = []

    for batch_images, batch_labels in tqdm(loader):
        batch_labels = batch_labels[:,0,:,:]
        batch_output = model(batch_images.to(device))
        batch_heatmaps = torch.sigmoid(batch_output[-1]['hm']).squeeze(dim=1).cpu()
        batch_gt = (batch_labels == 1)
        all_gt.extend(batch_gt)
        all_heatmaps.extend(batch_heatmaps)

    with open(f'eval_heatmaps_{arch}.pickle','wb') as f:
        pickle.dump(torch.stack(all_heatmaps), f)
    with open(f'eval_gt_{arch}.pickle','wb') as f:
        pickle.dump(torch.stack(all_gt), f)

def load_model_outputs(arch):

    with open(f'eval_heatmaps_{arch}.pickle','rb') as f:
        all_heatmaps = pickle.load(f)
    with open(f'eval_gt_{arch}.pickle','rb') as f:
        all_gt = pickle.load(f)

    return all_heatmaps, all_gt

def main(args=None):
    for arch in ['res_18','mobilenetv3small','dla_34']:
        # compute_model_outputs(arch=arch)
        all_heatmaps, all_gt = load_model_outputs(arch=arch)
        compute_precision_recall_hungarian(all_gt, all_heatmaps, enable_nms=True, output_filename=f'evaluation_{arch}')
        pr_curve_from_file(f'evaluation_{arch}.pickle')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
