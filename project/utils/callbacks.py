import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.model_summary import ModelSummary

import torch
import numpy as np
import wandb

from project.utils.labels import id2label


class ImageSegmentationLogger(Callback):
    def __init__(self, val_samples: torch.Tensor, num_samples: int = 32) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_ground_truth = val_samples

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_ground_truth = self.val_ground_truth.to(device=pl_module.device)

        # Get model prediction
        outputs = pl_module(val_imgs)
        preds = torch.argmax(outputs, 1)

        class_labels = dict((v.trainId, v.name) for (k, v) in id2label.items())

        for x, pred, y in zip(val_imgs[:self.num_samples],
                              preds[:self.num_samples],
                              val_ground_truth[:self.num_samples]):
            x = np.moveaxis(x.to(device='cpu').numpy(), 0, 2)
            pred = pred.to(device='cpu').numpy()
            y = y.to(device='cpu').squeeze().numpy()

            trainer.logger.experiment.log({
                "examples": wandb.Image(x, masks={
                    "prediction": {
                        "mask_data": pred,
                        "class_labels": class_labels
                    },
                    "ground truth": {
                        "mask_data": y,
                        "class_labels": class_labels
                    }
                })
            })


early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=False,
    mode='min'
)

MODEL_CKPT_PATH = 'model/'
MODEL_CKPT = 'model-{epoch:02d}-{val_loss:.2f}'

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=MODEL_CKPT_PATH,
    filename=MODEL_CKPT,
    save_top_k=3,
    mode='min')

summary_callback = ModelSummary(max_depth=-1)
