import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import numpy as np

from project.data.lightning_data_modules.city_scapes import CityScapesDataModule
from project.lightning_module.segmentation_module import SegmentationModule
from project.model.unet import UNet
from project.utils.callbacks import ImageSegmentationLogger, checkpoint_callback, early_stop_callback


def main():
    # TODO: Move to config
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Running on: " + str(device))

    wandb_logger = WandbLogger(
        project='gsn-roads-segmentation', job_type='train')

    model = SegmentationModule(UNet(3, 20))

    data_module = CityScapesDataModule()
    data_module.setup()
    val_samples = next(iter(data_module.val_dataloader()))

    trainer = pl.Trainer(max_epochs=2, accelerator=str(device), devices=1, logger=wandb_logger,
                         callbacks=[checkpoint_callback, early_stop_callback,
                         ImageSegmentationLogger(val_samples)])

    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
