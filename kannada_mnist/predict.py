from pathlib import Path

import lightning as L
import pandas as pd
import torch

from kannada_mnist.lightning_modules.infer_data_module import InferDataModule
from kannada_mnist.lightning_modules.module import KannadaMNISTModule
from kannada_mnist.models import get_model
from kannada_mnist.utilities.build_config import build_predict_config


def run_predict(cfg):
    infer_datamodule = InferDataModule(cfg)
    infer_datamodule.setup()

    model_instance = get_model(cfg.model, infer_datamodule)
    state_dict = torch.load(cfg.path_to_chkpt)

    module = KannadaMNISTModule.load_from_state_dict(
        state_dict=state_dict,
        model_instance=model_instance,
        datamodule=infer_datamodule,
        config=cfg,
    )
    module.eval()

    trainer = L.Trainer()
    predictions_list = trainer.predict(module, dataloaders=infer_datamodule.predict_dataloader())
    predictions = torch.cat(predictions_list, dim=0)

    output_file = Path(cfg.output_file)
    df = pd.DataFrame({"row_id": range(len(predictions)), "label": predictions.cpu().numpy()})

    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")


def predict(model: str, path_to_chkpt: str = None, output_file: str = None, input_file: str = None):
    cfg = build_predict_config(
        model=model, path_to_chkpt=path_to_chkpt, input_file=input_file, output_file=output_file
    )
    return run_predict(cfg)
