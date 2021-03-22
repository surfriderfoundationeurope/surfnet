import torch
from torch.nn import Conv2d
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torchvision
from collections import OrderedDict

class MySimpleSegmentationModel(Module):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier, num_classes=3, downsampling_factor=4, focal=True):
        super(MySimpleSegmentationModel, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self.downsampling_factor = downsampling_factor
        self.focal = focal
        if self.focal:
            self.classifier[-1] = Conv2d(self.classifier[-1].in_channels, self.num_classes+2, 1)
            if self.aux_classifier is not None:
                self.aux_classifier[-1] = Conv2d(self.aux_classifier[-1].in_channels, self.num_classes+2, 1)
        else:
            self.classifier[-1] = Conv2d(self.classifier[-1].in_channels, self.num_classes+1, 1)
            if self.aux_classifier is not None:
                self.aux_classifier[-1] = Conv2d(self.aux_classifier[-1].in_channels, self.num_classes+1, 1)



    def forward(self, x):
        input_shape = x.shape[-2:]
        downsampled_shape = (input_shape[0] // self.downsampling_factor, input_shape[1] // self.downsampling_factor)
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=downsampled_shape, mode='bilinear', align_corners=False)
        if self.focal:
            result["hm"] = x[:self.num_classes]
            result["wh"] = x[self.num_classes:]
        else:
            result["hm"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=downsampled_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result

def get_model(model_name, num_classes, freeze_backbone, downsampling_factor, focal=True):

    num_classes_pretrained = 21
    
    model_name_parts = model_name.split('__')

    classifier, backbone = model_name_parts[0], model_name_parts[1]

    if classifier  =='deeplabv3' and (backbone == 'resnet101' or backbone =='resnet50' or backbone == 'mobilenet_v3_large'):
        model = torchvision.models.segmentation.__dict__[classifier+'_'+backbone](num_classes=num_classes_pretrained,
                                                                aux_loss=False,
                                                                pretrained=True)
        # model = torch.hub.load('pytorch/vision:master',classifier+'_'+backbone)
    else:
        raise NotImplementedError

    model = MySimpleSegmentationModel(model.backbone, model.classifier, None, num_classes, downsampling_factor, focal)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model


if __name__ == "__main__":
    model = get_model('deeplabv3__resnet50', 3, freeze_backbone=False, downsampling_factor=4)
    print(model)


