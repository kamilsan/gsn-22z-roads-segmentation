import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.datasets import Cityscapes
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, PILToTensor


class CityScapesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 10, data_directory: str = './dataset', *kwargs):
        super().__init__()
        self.data_directory = data_directory
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        print('batch_size:', batch_size)
        self.batch_size = batch_size
        self.image_size = (572, 572)

        self.imagenet_transform = Compose([
            Resize(self.image_size),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = Compose([
            Resize((388, 388)),
            PILToTensor()
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Cityscapes(self.data_directory, split='train', mode='fine', target_type='semantic',
                                            transform=self.imagenet_transform, target_transform=self.target_transform)
            self.val_dataset = Cityscapes(self.data_directory, split='val', mode='fine', target_type='semantic',
                                          transform=self.imagenet_transform, target_transform=self.target_transform)
        if stage == 'test' or stage is None:
            self.test_dataset = Cityscapes(self.data_directory, split='test', mode='fine', target_type='semantic',
                                           transform=self.imagenet_transform, target_transform=self.target_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
