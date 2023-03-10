from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, PILToTensor, InterpolationMode

import pytorch_lightning as pl

from project.utils.labels import id2label
from project.utils.remap_labels import RemapCityscapesLabels


class CityScapesDataModule(pl.LightningDataModule):
    def __init__(self, image_size, norm_mean, norm_std, batch_size: int = 4, data_directory: str = './dataset',
                 num_workers: int = 4, **kwargs) -> None:
        super().__init__()
        self.data_directory = data_directory
        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size
        self.image_size = tuple(image_size)

        self.input_transform = Compose([
            Resize(self.image_size),
            ToTensor(),
            Normalize(mean=norm_mean, std=norm_std)
        ])

        self.target_transform = Compose([
            RemapCityscapesLabels(id2label),
            Resize(self.image_size,
                   interpolation=InterpolationMode.NEAREST),
            PILToTensor()
        ])

        self.num_workers = num_workers

    def setup(self, stage: str = None) -> None:
        if stage == 'fit' or stage is None:
            self.train_dataset = Cityscapes(self.data_directory, split='train', mode='fine', target_type='semantic',
                                            transform=self.input_transform, target_transform=self.target_transform)
            self.val_dataset = Cityscapes(self.data_directory, split='val', mode='fine', target_type='semantic',
                                          transform=self.input_transform, target_transform=self.target_transform)
        if stage == 'test' or stage is None:
            self.test_dataset = Cityscapes(self.data_directory, split='test', mode='fine', target_type='semantic',
                                           transform=self.input_transform, target_transform=self.target_transform)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
