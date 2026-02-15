import os

import torch

from kannada_mnist.models import get_model
from kannada_mnist.utilities.build_config import build_onnx_export_config
from kannada_mnist.utilities.constants import DEFAULT_BATCH_INPUT_SHAPE, DEFAULT_NUM_CLASSES


def run_export(model: str, input_file: str, output_file: str):
    model = get_model(model, num_classes=DEFAULT_NUM_CLASSES)
    model.load_state_dict(torch.load(input_file))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    torch.onnx.export(
        model,
        (torch.zeros(*DEFAULT_BATCH_INPUT_SHAPE),),
        output_file,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes={"input_tensor": (torch.export.Dim.DYNAMIC, None, None, None)},
        external_data=False,
    )

    print(f"ONNX model exported to {output_file}")


def export_to_onnx(model: str, path_to_chkpt: str = None, output_path: str = None):
    cfg = build_onnx_export_config(model=model, input_file=path_to_chkpt, output_file=output_path)
    run_export(**cfg)
