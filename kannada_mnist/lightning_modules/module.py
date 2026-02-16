import subprocess
from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchmetrics
from omegaconf import DictConfig

from kannada_mnist.utilities.constants import DEFAULT_PLOTS_DIR


class KannadaMNISTModule(L.LightningModule):
    def __init__(
        self,
        model,
        datamodule: L.LightningDataModule,
        config: DictConfig,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "datamodule"])
        self.model = model
        self.datamodule = datamodule
        self.config = config
        self.num_classes = datamodule.get_num_classes()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

        self.plots_dir = Path(DEFAULT_PLOTS_DIR)
        self.plots_dir.mkdir(exist_ok=True, parents=True)

        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def on_fit_start(self):
        if self.logger is not None:
            model_name = self.model.__class__.__name__
            git_commit = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            )
            self.logger.log_hyperparams(
                {
                    "model_type": model_name,
                    "git_commit": git_commit,
                    "optimizer_lr": self.config.optimizer.lr,
                    "scheduler_factor": self.config.scheduler.factor,
                    "scheduler_patience": self.config.scheduler.patience,
                }
            )
            self.logger.experiment.set_tag(
                key="Dataset", value="Kannada MNIST", run_id=self.logger.run_id
            )

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        # Useless plot
        # self.log(
        #     "lr",
        #     self.trainer.optimizers[0].param_groups[0]["lr"],
        #     on_step=False,
        #     on_epoch=True,
        # )

        return loss

    def on_train_epoch_end(self):
        avg_loss = self.trainer.callback_metrics.get("train_loss")
        if avg_loss is not None:
            self.train_losses.append(avg_loss.item())

    def on_train_end(self):
        if self.train_losses:
            self.visualize(
                title="Train Loss",
                label="train_loss",
                losses=self.train_losses,
                to_save=self.plots_dir / f"train_loss_{self.model.__class__.__name__}.png",
            )

        if self.val_losses:
            self.visualize(
                title="Validation Loss",
                label="val_loss",
                losses=self.val_losses,
                to_save=self.plots_dir / f"val_loss_{self.model.__class__.__name__}.png",
            )

        if self.val_accuracies:
            self.visualize(
                title="Validation Accuracy",
                label="val_accuracy",
                losses=self.val_accuracies,
                to_save=self.plots_dir / f"val_accuracy_{self.model.__class__.__name__}.png",
            )

    def validation_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)

        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.accuracy.update(outputs, targets)
        self.log(
            "val_accuracy", self.accuracy, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get("val_loss")
        val_acc = self.trainer.callback_metrics.get("val_accuracy")

        if val_loss is not None:
            self.val_losses.append(val_loss.item())

        if val_acc is not None:
            self.val_accuracies.append(val_acc.item())

    def test_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)

        self.accuracy.update(outputs, targets)
        self.log(
            "test_accuracy", self.accuracy, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )

    def on_predict_start(self):
        self._predict_ids = []
        self._predict_labels = []

    def predict_step(self, batch):
        inputs, ids = batch
        outputs = self.forward(inputs)
        predicted_labels = torch.argmax(outputs, dim=1)

        self._predict_ids.append(ids.detach().cpu())
        self._predict_labels.append(predicted_labels.detach().cpu())

        return predicted_labels

    def on_predict_epoch_end(self):
        ids = torch.cat(self._predict_ids)
        preds = torch.cat(self._predict_labels)

        output_file = Path(self.config.output_file)
        df = pd.DataFrame({"id": ids.numpy(), "label": preds.numpy()})

        df.to_csv(output_file, index=False)
        print(f"Saved predictions to {output_file}")

    def visualize(self, title: str, label: str, losses: list[float], to_save: Path):
        plt.figure()
        plt.plot(losses, label=label)
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend()
        plt.savefig(to_save)
        plt.close()

    def get_optimizer(self):
        cfg = self.config.optimizer
        return torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
        )

    def get_scheduler(self, optimizer):
        cfg = self.config.scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=cfg.mode,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
        return {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": cfg.monitor,
        }

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
