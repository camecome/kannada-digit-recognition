from hydra import compose, initialize

from kannada_mnist.utilities.constants import (
    CONFIG_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    INFERENCE_CONFIG_NAME,
    MODEL_REGISTRY,
    TRAINING_CONFIG_NAME,
)


def build_train_config(model: str, output_dir: str):
    overrides = []

    if model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {model}. Expected {MODEL_REGISTRY}")

    overrides.append(f"+model={model}")

    output_dir = (output_dir or DEFAULT_CHECKPOINT_DIR).rstrip("/")
    path_to_ckpt = f"{output_dir}/{model}.ckpt"

    overrides.append(f"+training.path_to_ckpt={path_to_ckpt}")

    with initialize(config_path=CONFIG_DIR, version_base=None):
        return compose(config_name=TRAINING_CONFIG_NAME, overrides=overrides)


def build_test_config(model: str, path_to_ckpt: str):
    overrides = []

    if model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {model}. Expected {MODEL_REGISTRY}")

    overrides.append(f"+model={model}")

    path_to_ckpt = path_to_ckpt or f"{DEFAULT_CHECKPOINT_DIR}/{model}.ckpt"
    overrides.append(f"+training.path_to_ckpt={path_to_ckpt}")

    with initialize(config_path=CONFIG_DIR, version_base=None):
        return compose(config_name=TRAINING_CONFIG_NAME, overrides=overrides)


def build_predict_config(model: str, path_to_ckpt: str, input_file: str, output_file: str):
    overrides = []

    if model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {model}. Expected {MODEL_REGISTRY}")

    overrides.append(f"+model={model}")

    path_to_ckpt = path_to_ckpt or f"{DEFAULT_CHECKPOINT_DIR}/{model}.ckpt"
    input_file = input_file or f"{DEFAULT_DATA_DIR}/predict.csv"
    output_file = output_file or f"{DEFAULT_DATA_DIR}/{model}_predictions.csv"

    overrides.append(f"+path_to_ckpt={path_to_ckpt}")
    overrides.append(f"+input_file={input_file}")
    overrides.append(f"+output_file={output_file}")

    with initialize(config_path=CONFIG_DIR, version_base=None):
        return compose(config_name=INFERENCE_CONFIG_NAME, overrides=overrides)


def build_onnx_export_config(model: str, path_to_ckpt: str, output_dir: str):
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {model}. Expected {MODEL_REGISTRY}")

    path_to_ckpt = path_to_ckpt or f"{DEFAULT_CHECKPOINT_DIR}/{model}.ckpt"
    output_dir = (output_dir or DEFAULT_CHECKPOINT_DIR).rstrip("/")
    output_file = f"{output_dir}/{model}_onnx.onnx"

    return {"model": model, "path_to_ckpt": path_to_ckpt, "output_file": output_file}
