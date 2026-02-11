import fire

from kannada_mnist.predict import predict
from kannada_mnist.test import test
from kannada_mnist.train import train

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "test": test,
            "predict": predict,
        }
    )
