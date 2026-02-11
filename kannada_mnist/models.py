import torch.nn as nn


# TODO: replace with a more complex model
class SimpleConv(nn.Module):
    """Simple convolutional model for image classification"""

    def __init__(self, num_classes: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, num_classes),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc(x)
        return x


class ConvClassifier(nn.Module):
    """Convolutional model for image classification"""

    def __init__(self, num_classes: int, dropout_conv: float = 0.25, dropout_fc: float = 0.4):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32, momentum=0.15),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout_conv),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=0.15),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout_conv),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32, momentum=0.15),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout_conv),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fc),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x
