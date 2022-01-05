"""
Ce script réalise la conversion d'un modèle Pytorch en Python vers un modèle Pytorch en TorchScript interprétable sur mobile.
"""


from tools.misc import load_model
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

from torch._C import MobileOptimizerType
from load_dataset import create_input_example


exemple_input = torch.load("models/exemple_input")

# Chargement du modèle : 
model= load_model("mobilenet_v3_small", None, "cpu")


#Fusion des couches (ne fonctionne pas pour les implémentations actuelles) : 
# model = torch.quantization.fuse_modules(
#     model, [['bn', 'relu']])


#Quantization : 
nb_exemples = 10 #à modifier pour choisir le nombre d'exemples utilisés dans la préparation

model.qconfig = torch.quantization.get_default_qconfig("qnnpack")

model_prepared = torch.quantization.prepare(model)
dataset_input = create_input_example(nb_exemples)
model_prepared(dataset_input)

model = torch.quantization.convert(model_prepared)
print("Quantization réalisée")


#Création du code TorchScript  (Script ou Trace ) : 

# traced_script_module = torch.jit.trace(model, example_inputs=exemple_input)
traced_script_module = torch.jit.script(model)

print("Création du code TorchScript Réalisé")


#Optmisations pour mobile : 
optimized_traced_model = optimize_for_mobile(traced_script_module , optimization_blocklist= {MobileOptimizerType.INSERT_FOLD_PREPACK_OPS })
# print("Optimisations réalisées")


#Sérialisation du modèle :
optimized_traced_model .save("models/S_MobileNet_SO5.pt")
print("Modele sauvegardé")

