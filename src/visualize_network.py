from detection.centernet.models import create_model




from torch.utils.tensorboard import SummaryWriter
from detection.coco_utils import get_surfrider
from detection.transforms import TrainTransforms  
from torch.utils.data import DataLoader
import torchvision.models as models
import torch 
import torch.nn as nn 
writer = SummaryWriter('experiments')
from time import time


if __name__ == '__main__':
    print(torch.__version__)

    transforms = TrainTransforms(540, (544,960), 1, 4)
    dataset = get_surfrider('data/images','train',transforms=transforms)
    loader = iter(DataLoader(dataset, shuffle=True, batch_size=16))

    model = create_model('res_18',heads={'hm':1},head_conv=256).to('cpu')
    # model = nn.Sequential(*list(model.children())[:-2]).to('cpu')
    model.eval()

    print(model)
    model.eval()
    # # print(model)
    # # model = create_model(arch='res_18', heads={'hm':1}, head_conv=256)
    # # backbone = nn.Sequential(*list(model.children())[:-2]).to('cpu')
    # images = loader.next()[0].to('cpu')

    # with torch.no_grad():
    #     time0 = time()
    #     print('Output shape:', model(images).shape)
    #     print(time() - time0)



