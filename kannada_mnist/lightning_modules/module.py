import lightning as L
import torch
import torchmetrics


class KannadaMNISTModule(L.LightningModule):
    """Module for training, evaluation and testing models for the classification task."""

    def __init__(self, model, datamodule: L.LightningDataModule):
        super().__init__()
        self.model = model
        self.datamodule = datamodule

        # Yet to get the number of classes from the somewhere, so for now hardcoding it to 10
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

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

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)

        self.accuracy.update(outputs, targets)
        self.log(
            "test_accuracy", self.accuracy, prog_bar=True, logger=True, on_step=False, on_epoch=True
        )

    def predict_step(self, batch, batch_idx):
        inputs = batch
        outputs = self.forward(inputs)
        predicted_labels = torch.argmax(outputs, dim=1)
        return predicted_labels

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        return optimizer

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-5
        )

        lr_scheduler = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val_accuracy",  # Assuming we have a validation metric named 'val_accuracy'
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
