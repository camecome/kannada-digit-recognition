from pathlib import Path

import lightning as L
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch import Generator
from torch.utils.data import DataLoader, Dataset, random_split

from kannada_mnist.utilities.tmp_kannada_mnist_dataset import KannadaMNISTDataset
from kannada_mnist.utilities.transforms import ToTensor28x28


class KannadaMNISTDataModule(L.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        data_dir = Path(config.data_dir or "data")
        self.train_path = data_dir / "train.csv"
        self.test_path = data_dir / "test.csv"

        self.batch_size = config.batch_size
        self.data_split_ratio = config.data_split_ratio
        self.num_workers = config.num_workers

        self._generator = Generator().manual_seed(config.random_seed)
        self._default_transform = transforms.Compose([ToTensor28x28()])

    def train_val_split(self, dataset: Dataset):
        val_size = int(len(dataset) * self.data_split_ratio)
        train_size = len(dataset) - val_size
        return random_split(dataset, [train_size, val_size], generator=self._generator)

    def setup(self, stage: str | None = None):
        stage_map = {
            "fit": (self.train_path, self._default_transform, True),
            "test": (self.test_path, self._default_transform, True),
        }

        if stage not in stage_map:
            raise ValueError(f"Invalid stage: {stage}. Expected one of {list(stage_map.keys())}")

        data_path, transform, has_labels = stage_map[stage]
        dataset = KannadaMNISTDataset(data_path, transform=transform, has_labels=has_labels)

        self.num_classes = dataset.get_num_classes() if has_labels else None

        if stage == "fit":
            self.custom_train_dataset, self.custom_val_dataset = self.train_val_split(dataset)
        else:
            setattr(self, f"{stage}_dataset", dataset)

    def _init_dataloader(self, dataset, batch_size, shuffle=False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
        )

    def get_num_classes(self):
        return self.num_classes

    def train_dataloader(self) -> DataLoader:
        return self._init_dataloader(self.custom_train_dataset, self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._init_dataloader(self.custom_val_dataset, self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return self._init_dataloader(self.test_dataset, self.batch_size)
