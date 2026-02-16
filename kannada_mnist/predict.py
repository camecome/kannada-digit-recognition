import lightning as L

from kannada_mnist.lightning_modules.infer_data_module import InferDataModule
from kannada_mnist.lightning_modules.module import KannadaMNISTModule
from kannada_mnist.models import get_model
from kannada_mnist.utilities.build_config import build_predict_config


def run_predict(cfg):
    infer_datamodule = InferDataModule(cfg)
    infer_datamodule.setup()

    model_instance = get_model(cfg.model, infer_datamodule)

    module = KannadaMNISTModule.load_from_checkpoint(
        checkpoint_path=cfg.path_to_ckpt,
        weights_only=False,
        model=model_instance,
        datamodule=infer_datamodule,
        config=cfg,
    )
    module.eval()

    trainer = L.Trainer()
    trainer.predict(module, dataloaders=infer_datamodule.predict_dataloader())


def predict(model: str, output_file: str = None, input_file: str = None, path_to_ckpt: str = None):
    cfg = build_predict_config(
        model=model, path_to_ckpt=path_to_ckpt, input_file=input_file, output_file=output_file
    )
    return run_predict(cfg)
