import numpy as np
import torch

from kannada_mnist.utilities.constants import IMAGE_H, IMAGE_W, PIXEL_MAX_VALUE


class ToTensor28x28:
    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def __call__(self, image: np.ndarray):
        image = image.reshape(IMAGE_H, IMAGE_W)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.normalize:
            image = image / PIXEL_MAX_VALUE

        return image
