import lightning as L
import torch
import torchmetrics
from omegaconf import DictConfig


class KannadaMNISTModule(L.LightningModule):
    """Module for training, evaluation and testing models for the classification task."""

    def __init__(
        self,
        model,
        datamodule: L.LightningDataModule,
        config: DictConfig,
    ):
        super().__init__()
        self.model = model
        self.datamodule = datamodule
        self.config = config

        self.num_classes = datamodule.get_num_classes()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)

        loss = self.criterion(outputs, targets)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        self.accuracy.update(outputs, targets)
        self.log(
            "val_accuracy", self.accuracy, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )

    def test_step(self, batch):
        inputs, targets = batch
        outputs = self.forward(inputs)

        self.accuracy.update(outputs, targets)
        self.log(
            "test_accuracy", self.accuracy, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )

    def predict_step(self, batch):
        inputs = batch
        outputs = self.forward(inputs)
        predicted_labels = torch.argmax(outputs, dim=1)
        return predicted_labels

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
