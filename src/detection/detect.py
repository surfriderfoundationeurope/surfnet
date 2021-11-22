import torch



def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def detect(preprocessed_frames, threshold, model):
    result = torch.sigmoid(model(preprocessed_frames)[-1]['hm'])
    detections = nms(result).gt(threshold).squeeze()
    return torch.nonzero(detections).cpu().numpy()[:, ::-1]
