import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class KannadaMNISTDataset(Dataset):
    def __init__(
        self, path_to_csv: str, transform: transforms.Compose = None, has_labels: bool = True
    ):
        self.df = pd.read_csv(path_to_csv)
        self.transform = transform
        self.has_labels = has_labels

        if self.has_labels:
            self.num_classes = self.df.iloc[:, 0].nunique()
        else:
            self.num_classes = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.has_labels:
            image = row[1:].values.astype(np.float32)
            label = torch.tensor(row[0], dtype=torch.long)
        else:
            image = row.values.astype(np.float32)
            label = None

        if self.transform:
            image = self.transform(image)

        if label is not None:
            return image, label

        return image
