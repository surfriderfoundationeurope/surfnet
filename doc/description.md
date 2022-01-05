# Plastic Origin

## Description des fichiers de code

src/convert.py : permet la conversion d'un modèle Pytorch de Python vers Torch Script.

src/detect.py : permet de produire des inférences sur une liste d'images.

scripts/detect.sh : permet de lancer detect.py. Basé sur track.sh.

src/load_dataset.py : contient la fonction ```create_input_example``` permettant de créer la liste d'exemple donnés au modèle lors de la quantization.

data/images/create_batch.py : script utilisé pour créer des batchs d'images servant aux tests de performance. 

## Descriptions des modèles 

models/S_MobileNet_noQ.pt : MobileNet optimisé pour le mobile et converti en Torch Script par ```script``` sans quantization.

models/S_MobileNet_10.pt : MobileNet optimisé pour le mobile et converti en Torch Script par ```script``` avec une quantization préparée par 10 exemples.

models/S_MobileNet_100.pt : MobileNet optimisé pour le mobile et converti en Torch Script par ```script``` avec une quantization préparée par 100 exemples.

models/S_MobileNet_100_R.pt : MobileNet optimisé pour le mobile et converti en Torch Script par ```script``` avec une quantization préparée par 100 exemples avec l'ajout d'un Tensor random pour gérer les cas d'égalité de score par la fonction ```nms```.

models/T_MobileNet_noQ : MobileNet optimisé pour le mobile et converti en Torch Script par ```trace``` sans quantization.

models/S_ResNet_noQ : ResNet18 optimisé pour le mobile et converti en Torch Script par ```script``` sans quantization.

models/T_ResNet_noQ : MobileNet optimisé pour le mobile et converti en Torch Script par ```trace``` sans quantization.

models/exemple_input : sauvegarde d'une image preprocessée issue du dataset SurfRider. Elle est utilisée pour permettre la conversion par ```trace```.

## Modifications des implémentations

Le fichier suivant a été modifié pour permettre la conversion par script et rajouter une partie de postprocessing au modèle MobileNet : 

```detection/centernet/networks/mobilenet.py```

```detection/centernet/networks/rmsra_resnet.py```

Les fichiers suivants ont été modifiés pour permettre la quantization du modèle MobileNet : 

```detection/centernet/networks/mobilenet.py```

```torchvision/models/mobilenetv3.py``` la version modifiée se trouve dans ```doc``` sous le nom ```mobilenetv3.py```.

```torchvision/ops/misc.py``` la version modifiée se trouve dans ```doc``` sous le nom ```misc.py```.



