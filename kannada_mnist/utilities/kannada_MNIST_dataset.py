from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class KannadaMNISTDataset(Dataset):
    def __init__(
        self, path_to_csv: Path, transform: transforms.Compose = None, has_labels: bool = True
    ):
        self.dataset = pd.read_csv(path_to_csv)
        self.transform = transform
        self.has_labels = has_labels

        if self.has_labels:
            self.num_classes = self.dataset.iloc[:, 0].nunique()
        else:
            self.num_classes = None

    def get_num_classes(self):
        return self.num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]

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
