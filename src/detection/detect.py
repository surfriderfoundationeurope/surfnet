import torch

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def detect(preprocessed_frames, threshold, model):
    batch_result = torch.sigmoid(model(preprocessed_frames)['hm'])
    batch_peaks = nms(batch_result).gt(threshold).squeeze(dim=1)
    detections = [torch.nonzero(peaks).cpu().numpy()[:,::-1] for peaks in batch_peaks]
    return detections
