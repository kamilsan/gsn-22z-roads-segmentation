import torch
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms

from lightning_data_modules.city_scapes import CityScapesDataModule
from models.unet import UNet
from lightning_modules.segmentation_module import SegmentationModule
from utils.callbacks import ImageSegmentationLogger, checkpoint_callback, early_stop_callback

import pytorch_lightning as pl


def main():
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Running on: " + str(DEVICE))

    wandb_logger = WandbLogger(project='gsn-roads-segmentation', job_type='train')

    model = SegmentationModule(UNet(3, 30))

    data_module = CityScapesDataModule()
    data_module.setup()
    val_samples = next(iter(data_module.val_dataloader()))

    trainer = pl.Trainer(max_epochs=3, accelerator=str(DEVICE), devices=1, logger=wandb_logger,
                         callbacks=[checkpoint_callback, early_stop_callback, ImageSegmentationLogger(val_samples)])

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
