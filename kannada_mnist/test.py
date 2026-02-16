import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

from kannada_mnist.lightning_modules.data_module import KannadaMNISTDataModule
from kannada_mnist.lightning_modules.module import KannadaMNISTModule
from kannada_mnist.models import get_model
from kannada_mnist.utilities.build_config import build_test_config


def run_testing(cfg):
    datamodule = KannadaMNISTDataModule(cfg.data)
    datamodule.setup(stage="test")

    model_instance = get_model(cfg.model, datamodule)

    module = KannadaMNISTModule.load_from_checkpoint(
        checkpoint_path=cfg.training.path_to_ckpt,
        weights_only=False,
        model=model_instance,
        datamodule=datamodule,
        config=cfg.training,
    )
    module.eval()

    mlflow_logger = MLFlowLogger(
        tracking_uri=cfg.logger.mlflow.tracking_uri,
        experiment_name=cfg.logger.mlflow.experiment_name,
    )

    trainer = L.Trainer(
        logger=mlflow_logger,
    )

    trainer.test(module, datamodule=datamodule)


def test(model: str, path_to_ckpt: str = None):
    cfg = build_test_config(model, path_to_ckpt)
    run_testing(cfg)
