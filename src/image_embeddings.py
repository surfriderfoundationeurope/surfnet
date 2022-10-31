import os

from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d

from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F

writer = SummaryWriter("experiments")
from tqdm import tqdm
import torch
from torch import nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from plasticorigins.tools.misc import load_model
from plasticorigins.detection.coco_utils import get_surfrider
from plasticorigins.detection.transforms import TrainTransforms, ValTransforms


def extract_features():
    device = torch.device("cuda")

    transforms = ValTransforms(540, (544, 960), 1, 4)
    dataset = get_surfrider("data/images", "val", transforms=transforms)
    loader = DataLoader(dataset, shuffle=False, batch_size=1)

    model = load_model("res_18", "models/res18_pretrained.pth", device=device)
    model = nn.Sequential(*list(model.children())[:-2])
    model.eval()

    for batch_nb, (batch_images, _) in tqdm(enumerate(loader)):
        torch.save(
            model(batch_images.to(device)).squeeze().cpu(),
            f"data/images/image_features_val/feature_{batch_nb}.pt",
        )


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features_path, device):

        self.features_path = features_path
        self.num_features = len(os.listdir(features_path))
        self.device = device

    def __getitem__(self, index):

        return torch.load(
            os.path.join(self.features_path, f"feature_{index}.pt"),
            map_location=self.device,
        )

    def __len__(self):
        return self.num_features


class AutoEncoder(pl.LightningModule):
    def __init__(self, model=None):
        super().__init__()
        # self.encoder = nn.Sequential(nn.Linear(in_features = 17*30*512, out_features = 17*30), ReLU())
        # self.decoder = nn.Sequential(nn.Linear(in_features = 17*30, out_features = 17*30*512), ReLU())
        if model is None:

            self.encoder = nn.Sequential(
                Conv2d(in_channels=512, out_channels=64, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=64, out_channels=8, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=8, out_channels=1, kernel_size=1),
                ReLU(),
            )

            self.decoder = nn.Sequential(
                Conv2d(in_channels=1, out_channels=8, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=8, out_channels=64, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=64, out_channels=512, kernel_size=1),
                ReLU(),
            )

            self.ae = nn.Sequential(self.encoder, self.decoder)
        elif model == "unet":
            self.ae = torch.hub.load(
                "mateuszbuda/brain-segmentation-pytorch",
                "unet",
                in_channels=512,
                out_channels=1,
                init_features=32,
                pretrained=False,
            )
            self.encoder = nn.Sequential(
                self.ae.encoder1,
                self.ae.pool1,
                self.ae.encoder2,
                self.ae.pool2,
                self.ae.encoder3,
                self.ae.pool3,
                self.ae.encoder4,
                self.ae.pool4,
                self.ae.bottleneck,
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.encoder(x)

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
        loss = F.mse_loss(x, self.ae(x))
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss


def compare_heatmaps_with_and_without_ae():
    device = torch.device("cuda")
    model = AutoEncoder()
    ae = model.load_from_checkpoint(
        "experiments/autoencoder/lightning_logs/version_9/checkpoints/epoch=369-step=9249.ckpt",
        map_location=device,
    )

    transforms = TrainTransforms(540, (544, 960), 1, 4)
    dataset = get_surfrider("data/images", "val", transforms=transforms)
    loader = DataLoader(dataset, shuffle=False, batch_size=1)
    model = load_model("res_18", "models/res18_pretrained.pth", device=device)

    full_model = nn.Sequential(
        *list(model.children())[:-2], ae.ae, *list(model.children())[-2:]
    ).to(device)
    full_model.eval()
    model.eval()

    with torch.no_grad():
        for batch_nb, (batch_images, _) in tqdm(enumerate(loader)):

            batch_output_without_ae = model(batch_images.to(device))
            batch_output_with_ae = full_model(batch_images.to(device))

            batch_heatmaps_without_ae = torch.sigmoid(
                batch_output_without_ae[-1]["hm"]
            ).squeeze(dim=1)[0]
            batch_output_with_ae = torch.sigmoid(batch_output_with_ae).squeeze(dim=1)[0]

            fig, (ax0, ax1) = plt.subplots(1, 2)
            ax0.imshow(
                batch_heatmaps_without_ae.cpu().numpy(), cmap="gray", vmin=0, vmax=1
            )
            ax1.imshow(batch_output_with_ae.cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            plt.show()


def train_ae():

    pl.seed_everything(1234)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    model = AutoEncoder(model="unet")
    trainer = pl.Trainer(
        gpus=1, check_val_every_n_epoch=5, default_root_dir="experiments/autoencoder"
    )

    train_dataset = FeatureDataset("data/images/image_features_train", device=device)
    val_dataset = FeatureDataset("data/images/image_features_val", device=device)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=64, pin_memory=True
    )
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=64)

    # trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    features = next(iter(train_dataloader))
    print(features.shape)


train_dataset = FeatureDataset(
    "data/images/image_features_train", device=torch.device("cpu")
)
batch_features = next(iter(train_dataset)).unsqueeze(0)

model = AutoEncoder(model="unet")
model.eval()

with torch.no_grad():
    output = model(batch_features)
    print(output.shape)


# train_ae()
