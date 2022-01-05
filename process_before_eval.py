import json
import numpy as np



with open('instances.json') as f:
  annot_dict = json.load(f)

with open('instances_val.json') as f:
  annot_dic = json.load(f)
images_=annot_dic['images']

with open('dict_name.json') as f: #dictionnaire avec l'id des images et les noms des fichiers d'image
  names_dict = json.load(f)

def arrondi(x):
    if x-int(x)>0.5 :
        return int(x)+1
    return int(x)

def java2heatmap(file, GPUmodel=False):  #prend les heatmaps générees par le modèle Java, remet en bon shape (150,150)
    with open(f'{file}.json') as f:
        predictions_dict=json.loads(f.read())

    all_heatmaps={}
    if GPUmodel :
        for image in images_ :
            img_name=image['file_name']
            all_heatmaps[img_name]= predictions_dict[img_name]
    else :
        for image in images_ :
            img_name=image['file_name']
            for img, name in names_dict.items():
                if name==img_name :
                    all_heatmaps[img_name]= predictions_dict[img]

    all_heatmaps = np.asarray(list(all_heatmaps.values()))
    all_heatmaps=np.reshape(all_heatmaps,(50,152,152))
    centers= np.argwhere(all_heatmaps)
    resized_heatmaps=np.zeros((50,150,150))
    for center in centers :
        new_center=[center[0], arrondi((150/152)*center[1]), arrondi((150/152)*center[2])]
        resized_heatmaps[new_center[0]][new_center[1]][new_center[2]]=all_heatmaps[center[0]][center[1]][center[2]]
    with open(f"{file}.npy","wb") as f :
        np.save(f, resized_heatmaps)


java2heatmap('heatmap', GPUmodel=True)

 