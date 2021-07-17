import torch
import torchmetrics
import torchvision
import pytorch_lightning as pl


class StreetImageModel(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float):
        super().__init__()
        self.save_hyperparameters()

        self._init_model()
        self._init_criterion()
        self._init_metrics()

    def _init_model(self):
        self.model = torchvision.models.resnet50(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        # redefine fully connected layer
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, self.hparams.num_classes),
            torch.nn.LogSoftmax(dim=1)
        )

    def _init_criterion(self):
        self.criterion = torch.nn.NLLLoss()

    def _init_metrics(self):
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.fc.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        _, preds = torch.max(outputs, 1)

        self.train_acc(preds, labels)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)

        loss = self.criterion(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        _, preds = torch.max(outputs, 1)

        self.val_acc(preds, labels)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        loss = self.criterion(outputs, labels)
        return loss
