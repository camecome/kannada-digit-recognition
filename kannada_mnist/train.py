import fire
import lightning as L
import torch

from kannada_mnist.lightning_modules.data_module import KannadaMNISTDataModule
from kannada_mnist.lightning_modules.module import KannadaMNISTModule
from kannada_mnist.models import get_model
from kannada_mnist.utilities.build_config import build_config


def run_training(cfg):
    datamodule = KannadaMNISTDataModule(cfg.data)
    datamodule.setup(stage="fit")

    model_instance = get_model(cfg.model.model_name, datamodule)

    module = KannadaMNISTModule(
        model=model_instance,
        datamodule=datamodule,
        config=cfg.training,
    )

    trainer = L.Trainer(
        max_epochs=cfg.training.num_epochs,
        # enable_progress_bar=True,
    )

    trainer.fit(module, datamodule=datamodule)
    torch.save(module.model.state_dict(), cfg.training.chkpt_path)


def train(model=None):
    cfg = build_config(model)
    run_training(cfg)


if __name__ == "__main__":
    fire.Fire({"train": train})
