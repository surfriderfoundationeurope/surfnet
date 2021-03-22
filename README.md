# Deeplabv3 x Surfrider 

This repo hosts code used to train various versions of DeeplabV3 with different backbones, on Surfrider data. It is a modified version of the official torchvision scripts as found in https://github.com/pytorch/vision/tree/master/references/segmentation adapted for the Surfrider datasets. Additional, a supplementary network dubbed Surfnet is can be trained on pairs of consecutive images to reduce artifacts in heatmaps.


## Installation 

```shell
conda env create --name surfnet --file environment.yml
conda activate surfnet 
```

## Training Deeplab

Use sripts contained in scipts_for_experiments/ to train the model. 

1. Set the data_path accordingly. If needed, use surfrider_to_coco_converter.py to convert original Surfrider annotations to COCO format. 
2. Set the batch_size according to your machine 
3. Set the model backbone as desired. Right now only MobileNetV3, ResNet101 and ResNet50 are supported. Syntax is 'deeplabv3__<backbone>'
4. Refer to parse_args() in train.py for details on other arguments.
5. Refer to official torchvision documentation (link above) for distributed multi-GPU training. 


## Training Surfnet 

TO WRITE

## Visualize results 

TO WRITE


