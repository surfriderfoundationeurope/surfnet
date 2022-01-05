# Conversion

Le fichier  ```convert.py``` réalise la conversion d'un modèle Pytorch de Python vers TorchScript.

## Chargement du modèle

Deux architectures de modèles sont disponibles pour la conversion : **MobileNet_V3** et **ResNet18**. 
Pour plus de détails sur les modifications d'implémentation réalisées voir ```implementation_model.py```. 

## Quantization

La quantization permet d'avoir les poids du modèle codés en 8 bits (au lieu de 32 bits).  La quantization doit permettre de réduire le temps d'inférence et l'espace mémoire nécessaire au modèle mais implique généralement une perte de précision.

Il existe 3 types de quantizations possibles dans Pytorch : static, dynamic et quantization aware training. 

**Quantization static** : quantization réalisée après le training. C'est la méthode qui génère la plus grande perte de précision. 

**Quantization dynamic** : quantization réalisée avant le training. 

**Quantization aware training** : training spécial adapté à la quantization avec modélisation des effets de la quantization. C'est la méthode qui génère la plus faible perte d'imprécision.

Le fichier ```convert.py``` implemente la **quantization static** : 

### Configuration

La fonction ```torch.quantization.get_default_qconfig``` permet de choisir le backend réalisant la quantization. Selon le système d'exploitation utilisé différents backends sont disponibles. Certaines couches ne sont pas quantizables par tous les backends.

Backend utilisé : ```qnnpack```

### Fusion de couches

Une optimisation classique réalisée avant la quantization est la fusion de couche. La fonction ```torch.quantization.fuse_modules``` permet de gérer manuellement la fusion des couches. Cependant cette optimisation est impossible sur les implémentations actuelles car les couches sont nommées par "block" et non par des noms indiquant leur nature. Il faut donc changer la convention de nommage. (cf doc ```torch.quantization.fuse_modules```)

### Preparation

La quantization **static** necessite la préparation du modèle pour déterminer les paramètres optimaux de quantization. Pour réaliser cette préparation on nourrit le modèle avec une série d'exemples.

Les exemples sont générés aléatoirement par la fonction ```create_input_example``` du fichier ```load_dataset.py``` à partir des images du dataset SurfRider.

Le comportement du modèle quantizé peut varier selon le nombre et la nature des exemples donnés.

### Problèmes rencontrés lors de la quantization : 

1. Opérations de base non quantizables : 

    Les couches quantizées ne peuvent travailler qu'avec des vecteurs quantizés. Il faut donc modifier l'implémentation du modèle en rajoutant une couche de quantification au début et une couche de dequantification à la fin. 

    Mais certaines opérations réalisées dans leur implémentations actuelles ne sont pas compatibles avec des vecteurs quantizés. Il s'agit en particulier des opérations * et + présentes dans 

    - ```torchvision/models/mobilenetv3.py```
    - ```torchvision/ops/misc.py```

    La solution implémentée est de rajouter des couches de quantization et de dequantization avant et après chacune de ses opérations. Mais cela se fait au prix d'un temps d'inférence plus important.

    Une autre solution possible serait des équivalents quantizables à ces opérations.

2. Disfonctionnement de la fonction nms : 

    Une erreur rencontrée sur les modèles quantizés est l'apparition de plusieurs centres sur le même déchet sur des pixels adjacents. Ce comportement est censé disparaitre grâce à la fonction ```nms``` en postprocessing mais celle ci gère mal le cas où deux pixels adjacents dans la heatmap ont exactement le même score. 

    Pour résoudre ce problème d'égalité des scores, on rajoute un bruit blanc statistique à la heatmap. 


## Création du code Torch Script
Il existe deux méthodes pour créer du code Torch Script : ```trace``` et ```script```.

1. ```trace``` : cela consiste à passer un exemple dans le modèle et enregistrer toutes les opérations réalisées sur les Tensors. Cette méthode fonctionne dans la plupart des cas mais a le défaut de ne pas rendre compte du control flow (par exemple les boucles if ou for). Ainsi pour certaines implémentations la conversion sera dépendantes de l'exemple utilisé.

2. ```script``` : cette fonction inspecte le code source du modèle et le compile en code TorchScript. Cette méthode conserve l'intégralité de la structure du modèle. Cependant tout les codes Python ne sont pas compilables, il est donc parfois necessaire de modifier l'implémentation du modèle (cf implémentation.md).

L'exemple utilisé pour la fonction ```trace``` est une image issue du dataset SurfRider et passée par le preprocessing. Elle est sauvegardée sous forme d'array numpy dans le fichier ```exemple_input```

Après modification de l'implémentation de **MobileNet** et **ResNet18**, les deux méthodes fonctionnent et sont disponibles dans le fichier ```convert.py```. 

## Optimisation pour mobile

La fonction ```torch.utils.mobile_optimizer.optimize_for_mobile``` réalise un ensemble d'optimisations classiques du modèle pour l'exécution sur mobile. 

Certaines optimisations ne sont pas compatibles avec tous les languages. En particulier, l'optimisation _"Insert and Fold prepacked ops"_ n'est pas compatible avec JAVA. Il est possible de retirer certaines optimisations en utilisant l'argument ```optimization_blocklist```. 


