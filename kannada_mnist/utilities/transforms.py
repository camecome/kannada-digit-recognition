import numpy as np
import torch


class ToTensor28x28:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def __call__(self, image: np.ndarray):
        image = image.reshape(28, 28)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.normalize:
            image = image / 255.0

        return image
