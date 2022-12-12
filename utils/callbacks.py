import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import wandb


class ImageSegmentationLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, 1)

        class_labels = dict(zip(range(1, 34), [str(x) for x in range(1, 34)]))

        for x, pred, y in zip(val_imgs[:self.num_samples],
                              preds[:self.num_samples],
                              val_labels[:self.num_samples]):

            pred = pred.to(device='cpu').numpy()
            y = y.to(device='cpu').squeeze().numpy()

            trainer.logger.experiment.log({
                "examples": wandb.Image(np.moveaxis(x.to(device='cpu').numpy(), 0, 2), masks={
                    "predictions": {
                        "mask_data": pred,
                        "class_labels": class_labels
                    },
                    "ground_truth": {
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
