import torch 
import torchvision.transforms as T 
import cv2 

def transform_for_test():

    transforms = []

    # transforms.append(ResizeForCenterNet(fix_res))
    transforms.append(T.Lambda(lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def detect(frame, threshold, model):

    frame = transform_for_test()(frame).to('cuda').unsqueeze(0)
    result = torch.sigmoid(model(frame)[-1]['hm'])
    detections = nms(result).gt(threshold).squeeze()

    return torch.nonzero(detections).cpu().numpy()[:, ::-1]