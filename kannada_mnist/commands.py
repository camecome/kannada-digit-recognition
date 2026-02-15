import fire

from kannada_mnist.predict import predict
from kannada_mnist.test import test
from kannada_mnist.train import train
from kannada_mnist.utilities.export_onnx import export_to_onnx
from kannada_mnist.utilities.generate_predict_dataset import generate_predict_dataset

if __name__ == "__main__":
    fire.Fire(
        {
            "train": train,
            "test": test,
            "predict": predict,
            "export_to_onnx": export_to_onnx,
            "generate_predict_dataset": generate_predict_dataset,
        }
    )
