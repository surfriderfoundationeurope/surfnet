from PIL import Image
import numpy as np
import os
from random import sample
from detection.transforms import TransformFrames
import torch

# Création du set d'exemples utilisés pour la préparration de la quantization du modèle
def create_input_example(number_input):
    list_images = [name for name in os.listdir('data/images/images')]
    frame_filenames = sample(list_images, k= number_input)

    dataset = []
    for frame_filename in frame_filenames: 
        print(frame_filename)
        frame = Image.open(os.path.join('data/images/images',frame_filename))
        frame = np.asarray(frame)
        if frame.shape[0] == 600:
            frame = TransformFrames()(frame)[None, :]
            dataset.append(frame)
    
    dataset = torch.cat(dataset)
    print(dataset.shape)
    return dataset

# create_input_example(10)