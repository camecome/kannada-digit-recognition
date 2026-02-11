import lightning as L
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from kannada_mnist.utilities.kannada_MNIST_dataset import KannadaMNISTDataset
from kannada_mnist.utilities.transforms import ToTensor28x28


class KannadaMNISTDataModule(L.LightningDataModule):
    _train_transforms = transforms.Compose([ToTensor28x28()])
    _val_transforms = transforms.Compose([ToTensor28x28()])
    _test_transforms = transforms.Compose([ToTensor28x28()])
    _predict_transforms = transforms.Compose([ToTensor28x28()])

    def __init__(self, data_dir: str, train_batch_size: int = 64, predict_batch_size: int = 64):
        super().__init__()

        self.train_data_path = f"{data_dir}/train.csv"
        self.val_data_path = f"{data_dir}/val.csv"
        self.test_data_path = f"{data_dir}/test.csv"
        self.predict_data_path = f"{data_dir}/predict.csv"

        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size

        self.num_workers = 0  # For MacOS compatibility

    def prepare_data(self):
        raise NotImplementedError(
            "Yet to implement the prepare_data method for KannadaMNISTDataModule"
        )

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_dataset = KannadaMNISTDataset(
                self.train_data_path, transform=self._train_transforms
            )
            self.val_dataset = KannadaMNISTDataset(
                self.val_data_path, transform=self._val_transforms
            )
        elif stage == "validate":
            self.val_dataset = KannadaMNISTDataset(
                self.val_data_path, transform=self._val_transforms
            )
        elif stage == "test":
            self.test_dataset = KannadaMNISTDataset(
                self.test_data_path, transform=self._test_transforms
            )
        elif stage == "predict":
            self.predict_dataset = KannadaMNISTDataset(
                self.predict_data_path, transform=self._predict_transforms, has_labels=False
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,  # For MacOS compatibility set to 0
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.predict_batch_size,  # We do not calculate gradients on validation set
            shuffle=False,
            num_workers=self.num_workers,  # For MacOS compatibility set to 0
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.predict_batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # For MacOS compatibility set to 0
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.predict_batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # For MacOS compatibility set to 0
        )
