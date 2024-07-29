from pytorch_lightning.core import LightningDataModule
from datasets import Shape16Dataset, ModelNet40Dataset
from torch.utils.data import DataLoader


class Shape16DataModule(LightningDataModule):
    def __init__(self, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(Shape16Dataset(mode='train', augment_flag=True), batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(Shape16Dataset(mode='val', augment_flag=False), batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, shuffle=False)
