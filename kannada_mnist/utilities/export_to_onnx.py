import os

import torch
from torch.export import Dim

from kannada_mnist.models import get_model
from kannada_mnist.utilities.build_config import build_onnx_export_config
from kannada_mnist.utilities.constants import DEFAULT_BATCH_INPUT_SHAPE, DEFAULT_NUM_CLASSES


def run_export(model: str, path_to_ckpt: str, output_file: str):
    model = get_model(model, num_classes=DEFAULT_NUM_CLASSES)
    model.eval()
    state_dict = torch.load(path_to_ckpt, weights_only=False)["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(*DEFAULT_BATCH_INPUT_SHAPE),),
        output_file,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"input_tensor": (Dim.DYNAMIC, Dim.STATIC, Dim.STATIC, Dim.STATIC)},
        external_data=False,
    )

    print(f"ONNX model exported to {output_file}")


def export_to_onnx(model: str, path_to_ckpt: str = None, output_dir: str = None):
    cfg = build_onnx_export_config(model=model, path_to_ckpt=path_to_ckpt, output_dir=output_dir)
    run_export(**cfg)
