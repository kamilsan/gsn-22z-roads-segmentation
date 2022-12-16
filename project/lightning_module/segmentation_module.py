from typing import Dict, Tuple

import hydra
import torch
import torch.nn.functional as F

import torchmetrics.functional as MF

import pytorch_lightning as pl


class SegmentationModule(pl.LightningModule):
    def __init__(self, model, optim, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.optimizer = optim
        self.num_classes = model.num_classes
        self.current_epoch_training_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(x, y)

    def common_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        outputs = self(x)
        loss = self.compute_loss(outputs, y.long().squeeze(dim=1))
        return loss, outputs, y

    def common_test_valid_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        loss, outputs, y = self.common_step(batch)
        preds = torch.argmax(outputs, dim=1)
        iou = MF.jaccard_index(preds.unsqueeze(
            dim=1), y, task="multiclass", num_classes=self.num_classes)
        return loss, iou

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, _, _ = self.common_step(batch)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def training_epoch_end(self, outs: torch.Tensor) -> None:
        self.current_epoch_training_loss = torch.stack(
            [o["loss"] for o in outs]).mean()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:
        loss, iou = self.common_test_valid_step(batch)
        self.log('val_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('val_IoU', iou, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_IoU': iou}

    def validation_epoch_end(self, outs: torch.Tensor) -> None:
        avg_loss = torch.stack([o["val_loss"] for o in outs]).mean()
        self.log('train and val losses', {
                 'train': self.current_epoch_training_loss.item(), 'val': avg_loss.item()})

    def test_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        loss, iou = self.common_test_valid_step(batch)
        self.log('test_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('test_IoU', iou, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_IoU': iou}

    def configure_optimizers(self) -> list:
        optimizer = hydra.utils.instantiate(
            self.hparams.optim, params=self.parameters())
        return [optimizer]
