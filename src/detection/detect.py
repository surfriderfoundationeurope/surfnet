import torch
from time import time
import numpy as np
from random import random
def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def detect(preprocessed_frames, threshold, model):
    # print("Shape : ", preprocessed_frames.shape)
    # batch_result = torch.sigmoid(model(preprocessed_frames)[-1]['hm'])
    print("oko ok ok oko")
    print(preprocessed_frames.shape)
    print(preprocessed_frames)
    batch_peaks = ((model(preprocessed_frames)))
    # print("avant : ", batch_peaks[0][0][87:92,67:70])
    # for i in range(len(batch_peaks[0][0])):
    #     for j in range((len(batch_peaks[0][0][0]))):
    #         batch_peaks[0][0][i,j] += random()*0.000001
    # batch_peaks = nms(batch_peaks)
    
    # print("après : ", batch_peaks[0][0][96:98,80:85])
    # print(batch_peaks[0][0][96,80].double())
    # print(float(batch_peaks[0][0][96,81]))
    # print(float(batch_peaks[0][0][96,82]))
    # print(float(batch_peaks[0][0][96,83]))
    # print(batch_peaks[0][0].numpy())

    # batch_peaks = batch_peaks.gt(threshold).squeeze(dim=1)
    # batch_peaks = [torch.nonzero(peaks).cpu().numpy()[:,::-1] for peaks in batch_peaks]
    return batch_peaks
