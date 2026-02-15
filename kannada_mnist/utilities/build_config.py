from hydra import compose, initialize

from kannada_mnist.utilities.constants import (
    CONFIG_DIR,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    INFERENCE_CONFIG_NAME,
    MODEL_REGISTRY,
    TRAINING_CONFIG_NAME,
)


def build_train_config(model: str, target_dir: str):
    overrides = []

    if model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {model}. Expected {MODEL_REGISTRY}")

    overrides.append(f"+model={model}")

    target_dir = (target_dir or DEFAULT_CHECKPOINT_DIR).rstrip("/")
    path_to_chkpt = f"{target_dir}/{model}.pth"

    overrides.append(f"+training.chkpt_path={path_to_chkpt}")

    with initialize(config_path=CONFIG_DIR, version_base=None):
        return compose(config_name=TRAINING_CONFIG_NAME, overrides=overrides)


def build_test_config(model: str, path_to_chkpt: str):
    overrides = []

    if model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {model}. Expected {MODEL_REGISTRY}")

    overrides.append(f"+model={model}")

    path_to_chkpt = path_to_chkpt or f"{DEFAULT_CHECKPOINT_DIR}/{model}.pth"
    overrides.append(f"+training.chkpt_path={path_to_chkpt}")

    with initialize(config_path=CONFIG_DIR, version_base=None):
        return compose(config_name=TRAINING_CONFIG_NAME, overrides=overrides)


def build_predict_config(model: str, path_to_chkpt: str, input_file: str, output_file: str):
    overrides = []

    if model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {model}. Expected {MODEL_REGISTRY}")

    overrides.append(f"+model={model}")

    path_to_chkpt = path_to_chkpt or f"{DEFAULT_CHECKPOINT_DIR}/{model}.pth"
    input_file = input_file or f"{DEFAULT_DATA_DIR}/predict.csv"
    output_file = output_file or f"{DEFAULT_DATA_DIR}/{model}_predictions.csv"

    overrides.append(f"+path_to_chkpt={path_to_chkpt}")
    overrides.append(f"+input_file={input_file}")
    overrides.append(f"+output_file={output_file}")

    with initialize(config_path=CONFIG_DIR, version_base=None):
        return compose(config_name=INFERENCE_CONFIG_NAME, overrides=overrides)


def build_onnx_export_config(model: str, input_file: str, output_file: str):
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Invalid model: {model}. Expected {MODEL_REGISTRY}")

    input_file = input_file or f"{DEFAULT_CHECKPOINT_DIR}/{model}.pth"
    output_file = output_file or f"{DEFAULT_CHECKPOINT_DIR}/{model}_onnx.onnx"

    return {"model": model, "input_file": input_file, "output_file": output_file}
