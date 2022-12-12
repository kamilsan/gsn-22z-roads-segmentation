import torch
import pytorch_lightning as pl

import torch.nn.functional as F
import torchmetrics.functional as MF

from torchvision.transforms import Resize, InterpolationMode


class SegmentationModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model_output_mask_size = None
        self.model = model
        self.lr = 0.02
        self.num_classes = 34
        self.current_epoch_training_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(x, y)

    def common_step(self, batch: torch.Tensor, batch_idx) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        x, y = batch
        outputs = self(x)
        self.model_output_mask_size = outputs.size()[-2:-1]
        outputs = Resize(x.size()[-2:-1], interpolation=InterpolationMode.NEAREST).forward(outputs)
        loss = self.compute_loss(outputs, y.long().squeeze())
        return loss, outputs, y

    def common_test_valid_step(self, batch, batch_idx):
        loss, outputs, y = self.common_step(batch, batch_idx)
        preds = torch.argmax(outputs, dim=1)
        IoU = MF.jaccard_index(preds.unsqueeze(dim=1), y, task="multiclass", num_classes=self.num_classes)
        return loss, IoU

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def training_epoch_end(self, outs):
        self.current_epoch_training_loss = torch.stack([o["loss"] for o in outs]).mean()

    def validation_step(self, batch, batch_idx):
        loss, IoU = self.common_test_valid_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_IoU', IoU, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_IoU': IoU}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([o["val_loss"] for o in outs]).mean()
        self.log('train and val losses', {'train': self.current_epoch_training_loss.item(), 'val': avg_loss.item()})

    def test_step(self, batch, batch_idx):
        loss, IoU = self.common_test_valid_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_IoU', IoU, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_IoU': IoU}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        return [optimizer], [lr_scheduler]
