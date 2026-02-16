import fire

from kannada_mnist.predict import predict
from kannada_mnist.test import test
from kannada_mnist.train import train
from kannada_mnist.utilities.export_to_onnx import export_to_onnx

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "test": test,
            "predict": predict,
            "export_to_onnx": export_to_onnx,
        }
    )
