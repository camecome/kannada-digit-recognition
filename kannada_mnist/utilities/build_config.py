from hydra import compose, initialize

CONFIG_PATH = "../configs"
CONFIG_NAME = "config.yaml"

MODEL_MAPPING = {
    "complex": "conv_classifier",
    "simple": "simple_classifier",
}


def build_config(model_alias: str | None):
    overrides = []

    if model_alias is not None:
        try:
            model_name = MODEL_MAPPING[model_alias]
            overrides.append(f"model.model_name={model_name}")
        except KeyError:
            raise ValueError(f"Invalid model: {model_alias}. Expected {list(MODEL_MAPPING)}")

    with initialize(config_path=CONFIG_PATH, version_base=None):
        return compose(config_name=CONFIG_NAME, overrides=overrides)
