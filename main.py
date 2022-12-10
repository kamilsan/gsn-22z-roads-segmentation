import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms

from lightning_data_modules.city_scapes import CityScapesDataModule
from models.unet import UNet
from lightning_modules.segmentation_module import SegmentationModule

import pytorch_lightning as pl


def main():
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Running on: " + str(DEVICE))

    model = SegmentationModule(UNet(3, 30))

    data_module = CityScapesDataModule()
    data_module.setup()
    # trainer = pl.Trainer(max_epochs=50, accelerator='gpu', devices=1, logger=wandb_logger,
    #                      callbacks=[checkpoint_callback, early_stop_callback, ImagePredictionLogger(val_samples)])

    trainer = pl.Trainer(max_epochs=50, accelerator=str(DEVICE), devices=1)

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
