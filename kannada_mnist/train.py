import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

from kannada_mnist.lightning_modules.data_module import KannadaMNISTDataModule
from kannada_mnist.lightning_modules.module import KannadaMNISTModule
from kannada_mnist.models import get_model
from kannada_mnist.utilities.build_config import build_train_config


def run_training(cfg):
    datamodule = KannadaMNISTDataModule(cfg.data)
    datamodule.setup(stage="fit")

    model_instance = get_model(cfg.model, datamodule)

    module = KannadaMNISTModule(
        model=model_instance,
        datamodule=datamodule,
        config=cfg.training,
    )

    mlflow_logger = MLFlowLogger(
        tracking_uri=cfg.logger.mlflow.tracking_uri,
        experiment_name=cfg.logger.mlflow.experiment_name,
    )

    trainer = L.Trainer(
        max_epochs=cfg.training.num_epochs,
        logger=mlflow_logger,
        log_every_n_steps=cfg.logger.mlflow.log_every_n_steps,
    )

    trainer.fit(module, datamodule=datamodule)

    # os.makedirs(os.path.dirname(cfg.training.path_to_ckpt), exist_ok=True)
    trainer.save_checkpoint(cfg.training.path_to_ckpt, weights_only=False)

    # torch.save(module.model.state_dict(), cfg.training.path_to_ckpt)


def train(model: str, output_dir: str = None):
    cfg = build_train_config(model=model, output_dir=output_dir)
    run_training(cfg)
