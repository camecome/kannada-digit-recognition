# Config files
CONFIG_DIR = "../../configs"
TRAINING_CONFIG_NAME = "training_config.yaml"
INFERENCE_CONFIG_NAME = "inference_config.yaml"

# Directories
DEFAULT_CHECKPOINT_DIR = "models"
DEFAULT_DATA_DIR = "data"
DEFAULT_PLOTS_DIR = "plots"

MODEL_REGISTRY = [
    "dummy_classifier",
    "conv_classifier",
]

# Image / data parameters
IMAGE_W, IMAGE_H = 28, 28
PIXEL_MAX_VALUE = 255.0
DEFAULT_BATCH_INPUT_SHAPE = (1, 1, IMAGE_W, IMAGE_H)
DEFAULT_NUM_CLASSES = 10
