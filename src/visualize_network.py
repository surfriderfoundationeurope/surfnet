from pytorch_lightning.accelerators import accelerator
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.loss import MSELoss
from torchvision.models.densenet import _load_state_dict
from tools.misc import load_model
import os 

from torch.utils.tensorboard import SummaryWriter
from detection.coco_utils import get_surfrider
from detection.transforms import TrainTransforms  
from torch.utils.data import DataLoader
import torchvision.models as models
import torch 
import torch.nn as nn 
import torch.nn.functional as F
writer = SummaryWriter('experiments')
from time import time
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import random_split
import pytorch_lightning as pl

import matplotlib.pyplot as plt 

    
def extract_features():
    device = torch.device('cuda')

    transforms = TrainTransforms(540, (544,960), 1, 4)
    dataset = get_surfrider('data/images','val',transforms=transforms)
    loader = DataLoader(dataset, shuffle=False, batch_size=1)

    model = load_model('res_18','models/res18_pretrained.pth',device=device)
    model = nn.Sequential(*list(model.children())[:-2])
    model.eval()

    for batch_nb, (batch_images, _) in tqdm(enumerate(loader)): 
        torch.save(model(batch_images.to(device)).squeeze().cpu(), f'data/images/image_features_val/feature_{batch_nb}.pt')

class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features_path, device):

        self.features_path = features_path
        self.num_features = len(os.listdir(features_path))
        self.device = device    

    def __getitem__(self, index):

        return torch.load(os.path.join(self.features_path,f'feature_{index}.pt'), map_location=self.device)

    def __len__(self):
        return self.num_features

class AutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # self.encoder = nn.Sequential(nn.Linear(in_features = 17*30*512, out_features = 17*30), ReLU())
        # self.decoder = nn.Sequential(nn.Linear(in_features = 17*30, out_features = 17*30*512), ReLU())

        self.encoder = nn.Sequential(
            Conv2d(in_channels=512, out_channels=64, kernel_size=1), 
            ReLU(),
            Conv2d(in_channels=64, out_channels=8, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=8, out_channels=1, kernel_size=1),
            ReLU()
        )


        self.decoder = nn.Sequential(
            Conv2d(in_channels=1, out_channels=8, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=8, out_channels=64, kernel_size=1),
            ReLU(),
            Conv2d(in_channels=64, out_channels=512, kernel_size=1),
            ReLU()
        )


    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _prepare_batch(self, batch):

        return batch

    def _common_step(self, batch, batch_idx, stage: str):
        x = self._prepare_batch(batch)
        loss = F.mse_loss(x, self(x))
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss

device = torch.device('cuda')
model = AutoEncoder()
ae = model.load_from_checkpoint('experiments/autoencoder/lightning_logs/version_9/checkpoints/epoch=369-step=9249.ckpt', map_location=device)

transforms = TrainTransforms(540, (544,960), 1, 4)
dataset = get_surfrider('data/images','val',transforms=transforms)
loader = DataLoader(dataset, shuffle=False, batch_size=1)
model = load_model('res_18','models/res18_pretrained.pth',device=device)

full_model = nn.Sequential(*list(model.children())[:-2], ae.encoder, ae.decoder, *list(model.children())[-2:]).to(device)
full_model.eval()
model.eval()

with torch.no_grad():
    for batch_nb, (batch_images, _) in tqdm(enumerate(loader)): 

        batch_output_without_ae = model(batch_images.to(device))
        batch_output_with_ae = full_model(batch_images.to(device))

        batch_heatmaps_without_ae = torch.sigmoid(batch_output_without_ae[-1]['hm']).squeeze(dim=1)[0]
        batch_output_with_ae = torch.sigmoid(batch_output_with_ae).squeeze(dim=1)[0]

        fig, (ax0, ax1) = plt.subplots(1,2)
        ax0.imshow(batch_heatmaps_without_ae.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax1.imshow(batch_output_with_ae.cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        plt.show()







    # pl.seed_everything(1234)
    # device = torch.device('cuda')
    # model = AutoEncoder()
    # trainer = pl.Trainer(gpus=2,
    #                      check_val_every_n_epoch=5, 
    #                      default_root_dir='experiments/autoencoder')

    # train_dataset = FeatureDataset('data/images/image_features_train', device=device)
    # val_dataset = FeatureDataset('data/images/image_features_val', device=device)

    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64)
    # val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=64)

    # trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)



