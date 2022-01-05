# Modifications réalisées sur l'implémentation des modèles Pytorch

## Modifications liées au postprocessing

Pour faciliter l'implémentation sur l'app démo en java, on inclue une partie du post processing directement dans le modèle Pytorch après la dernière couche. D'autant que pour permettre la conversion du modèle vers Torch Script il est necessaire de renvoyer un tensor.

En pratique, on modifie le forward en faisant passer l'output de la dernière couche par une sigmoid et la fonction nms.

## Modifications liées à la conversion par script

Pour permettre la conversion du modèle par la fonction ```script```, il est necessaire de rendre compatible l'implémentation en Python du modèle.

Exemples d'erreurs rencontrées :

1. Problème de typage : 

Il est souvent nécessaire de préciser le type de chaque argument de chaque fonction définie dans le modèle Pytorch. Sans typage le compilateur suppose que l'argument est de type Tensor. 

Il est utile d'utiliser le module ```typing``` pour typer les fonctions.

La liste des types compatibles est disponible sur https://pytorch.org/docs/stable/jit_language_reference.html .

2. Ambiguité du code : 

Certaines possibilités offertes par Python ne sont pas compatibles avec le compilateur Torch Script. C'est généralement le cas lorsque le code est non explicite, par exemple l'opérateur * pour unpack un nombre inconnu d'argument à partir d'une liste n'est généralement pas compatible. 

Il est necessaire d'avoir un code statique. 

3. Variabilité des types selon les executions : 

Certaines fonctions peuvent renvoyer des résultats avec des types différents selon les arguments passés en entré. Ce comportement n'est pas accepté par le compilateur Torch Script. 

C'est par exemple le cas dans l'implémentation du MobileNet avec la fonction ```__getattr__``` . En effet, cette fonction est utilisée pour obtenir les attributs de la classe et ces attributs sont de types différents. Ce comportement n'est pas accepté par le compilateur Torch Script.

Il a donc été necessaire de modifier l'implémentation des couches obtenues par l'appel de cette fonction. Pour cela on crée une nouvelle fonction ```_makeLayer_``` qui implémente le modèle séquentiel de la couche. Puis on définie un nouvelle attribut de la classe ```hm``` qui contient une instance de la couche. La couche est ensuite appelée normalement danns le forward. 

Chaque étape de cette nouvelle implémentation a été réalisé dans l'objectif de modifier l'implémentation de la dernière couche sans modifier la structure du modèle pour pouvoir toujours faire le lien avec les poids déjà entrainés du modèle. En particulier le choix du nom de l'attribut qui contient l'instance de la classe est importante car il détermine le nom de la couche dans la structure du moddèle.

Attention lors des modifications de l'implémentation des modèles il est possible que les noms de certaines couches soient modifiées. Il est donc nécessaire de relier les poids entrainés aux nouvelles couches OU ré-entrainer le modèle avec la nouvelle implémenntation.

## Modifications liées à la quantization

### Quantization des tensors

Les couches quantizés ne peuvent prendre que des Tensors quantizé en entrée. Il est donc nécessaire de rajouter une couche de quantization avant la première couche et une couche de dequantization après la dernière couche.

Ces couches sont définies comme attributs de classe par :          
```Python
self.quant = torch.quantization.QuantStub()
self.dequant = torch.quantization.DeQuantStub()
```

Ils sont ensuite rajoutés en début et en fin de forward.

### Opérations non compatibles

Certaines opérations de base ne sont pas compatibles avec la quantization. En particulier il s'agit des opérationns ```*``` et ```+```.

Dans l'implémentation actuelle de MobileNet, ces opérations sont présentes dans 3 fichiers :

- ```detection/centernet/networks/mobilenet.py```
- ```torchvision/models/mobilenetv3.py```
- ```torchvision/ops/misc.py```

Pour résoudre ce problème nous avons définis les couches de quantization et de dequantization comme attribut des modèles comme précedemment mentionné. Puis on rajoute une couche de dedquantization avant l'opération et une couche de quantization après. Ainsi l'opération n'a pas besoin d'être quantizée car elle travaille avec des vecteurs non quantizés. 

-> Un désavantage de cette solution est le temps d'inférence et la perte de performance causé par l'ajout de ses couches de quantization / dequantization.

Une autre solution pourait être de trouver un équivalent à ces opérations qui soit quantizable.


### Disfonctionnement de NMS

Une erreur rencontrée sur les modèles quantizés est l'apparition de plusieurs centres sur le même déchet sur des pixels adjacents. Ce comportement est sensé disparaitre grâce à la fonction ```nms``` en postprocessing mais celle ci gère mal le cas où deux pixels adjacents dans la heatmap ont exactement le même score. 

Pour résoudre ce problème d'égalité des scores, on rajoute un bruit blanc statistique à la heatmap. 

En pratique on ajoute une heatmap aléatoire créée par la fonction ```torch.rand``` dans la fonction ```nms``` directement implémentée dans le modèle.
 

