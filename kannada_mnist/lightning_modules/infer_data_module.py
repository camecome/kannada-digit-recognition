from pathlib import Path

import lightning as L
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from kannada_mnist.utilities.kannada_mnist_dataset import KannadaMNISTDataset
from kannada_mnist.utilities.transforms import ToTensor28x28


class InferDataModule(L.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.input_file = Path(config.input_file)
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.num_classes = config.num_classes

        self._default_transform = transforms.Compose([ToTensor28x28()])

    def get_num_classes(self):
        return self.num_classes

    def setup(self, stage: str | None = None):
        self.predict_dataset = KannadaMNISTDataset(
            self.input_file,
            transform=self._default_transform,
            has_labels=False,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
