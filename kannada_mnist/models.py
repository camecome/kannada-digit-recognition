import torch.nn as nn
from torch import Tensor

from kannada_mnist.utilities.constants import MODEL_REGISTRY


class DummyClassifier(nn.Module):
    """Simple convolutional model for image classification"""

    def __init__(self, num_classes: int):
        super().__init__()

        self.convolutional_block = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=16 * 14 * 14,
                out_features=num_classes,
            ),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        features = self.convolutional_block(input_tensor)
        logits = self.classifier(features)
        return logits


class ConvClassifier(nn.Module):
    """Convolutional model for image classification"""

    def __init__(self, num_classes: int):
        super().__init__()

        self.convolutional_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32, momentum=0.15),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
        )

        self.convolutional_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=64, momentum=0.15),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
        )

        self.convolutional_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=32, momentum=0.15),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=32 * 3 * 3,
                out_features=256,
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(
                in_features=256,
                out_features=num_classes,
            ),
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        features = self.convolutional_block_1(input_tensor)
        features = self.convolutional_block_2(features)
        features = self.convolutional_block_3(features)
        logits = self.classifier(features)
        return logits


# No way to get rid of this magic constants without making the code more complex.
# This is the best we can do to avoid circular imports.
MODELS_TO_CLASS = {
    "dummy_classifier": DummyClassifier,
    "conv_classifier": ConvClassifier,
}


def get_model(model: str, datamodule=None, num_classes: int = None):
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {model}. Expected {list(MODEL_REGISTRY.keys())}")

    model_cls = MODELS_TO_CLASS[model]
    if num_classes is not None:
        return model_cls(num_classes=num_classes)
    return model_cls(num_classes=datamodule.get_num_classes())
