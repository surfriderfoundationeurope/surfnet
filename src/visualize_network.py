from tools.misc import load_model
from detection.detect import nms




from torch.utils.tensorboard import SummaryWriter
from detection.coco_utils import get_surfrider
from detection.transforms import TrainTransforms  
from torch.utils.data import DataLoader
import torchvision.models as models
import torch 
import torch.nn as nn 
writer = SummaryWriter('experiments')
from time import time
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment



threshold = 0.3
def main(args=None):
    device = torch.device('cuda')

    transforms = TrainTransforms(540, (544,960), 1, 4)
    dataset = get_surfrider('data/images','val',transforms=transforms)
    loader = DataLoader(dataset, shuffle=False, batch_size=1)

    model = load_model('res_18','models/res18_pretrained.pth',device=device)
    model.eval()


    all_detections = []
    all_ground_truths = []

    for batch_images, batch_labels in tqdm(loader): 
        batch_labels = batch_labels[:,0,:,:]
        batch_output = model(batch_images)
        batch_peaks = nms(torch.sigmoid(batch_output[-1]['hm'])).gt(threshold).squeeze(dim=1)
        detections = [torch.nonzero(peaks).cpu().numpy()[:,::-1] for peaks in batch_peaks] 
        ground_truth = [torch.nonzero(labels == 1).numpy()[:,::-1] for labels in batch_labels] 
        all_detections.extend(detections)
        all_ground_truths.extend(ground_truth)




    






if __name__ == '__main__':

    
    main()

