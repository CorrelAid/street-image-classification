from typing import Tuple

import torch
import torchmetrics
import torchvision
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf
import pytorch_lightning as pl


class MTLMobileNetV3(torchvision.models.MobileNetV3):
    def __init__(self, inverted_residual_setting, last_channel):
        super().__init__(inverted_residual_setting, last_channel)

        self.last_channel = last_channel

        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        self.lastconv_output_channels = 6 * lastconv_input_channels

    def add_mtl_head(self, num_classes: int):
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.lastconv_output_channels, self.last_channel),
            torch.nn.Hardswish(inplace=True),
            torch.nn.Dropout(p=0.2, inplace=True)
        )
        self.classifier1 = torch.nn.Sequential(
            torch.nn.Linear(self.last_channel, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
        self.classifier2 = torch.nn.Sequential(
            torch.nn.Linear(self.last_channel, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )

    def _forward_impl(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        return x1, x2


def load_pretrained_mtl_mobilenet_v3_large(num_classes: int) -> MTLMobileNetV3:
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch)

    model = MTLMobileNetV3(inverted_residual_setting, last_channel)
    state_dict = torchvision.models.mobilenetv3.load_state_dict_from_url(
        torchvision.models.mobilenetv3.model_urls[arch], progress=True)
    model.load_state_dict(state_dict)
    model.add_mtl_head(num_classes=num_classes)
    return model


def get_metric_dict(mode: str, num_classes: int) -> dict:
    kwargs = {"num_classes": num_classes, "average": "weighted"}
    metric_dict = {
        f"accuracy_{mode}_surface": torchmetrics.Accuracy(**kwargs),
        f"accuracy_{mode}_smoothness": torchmetrics.Accuracy(**kwargs),
        f"precision_{mode}_surface": torchmetrics.Precision(**kwargs),
        f"precision_{mode}_smoothness": torchmetrics.Precision(**kwargs),
        f"f1_{mode}_surface": torchmetrics.F1(**kwargs),
        f"f1_{mode}_smoothness": torchmetrics.F1(**kwargs),
    }
    return metric_dict


class CargoRocketModel(pl.LightningModule):
    def __init__(self, num_classes: int = 3, learning_rate: float = 1e-3):
        super().__init__()
        self.num_classes = num_classes
        self.model = load_pretrained_mtl_mobilenet_v3_large(num_classes=self.num_classes)
        self.criterion1 = torch.nn.NLLLoss()
        self.criterion2 = torch.nn.NLLLoss()
        self.learning_rate = learning_rate

        self.train_metrics = get_metric_dict(mode="train", num_classes=self.num_classes)
        self.val_metrics = get_metric_dict(mode="val", num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y1 = batch["surface"]
        y2 = batch["smoothness"]

        y_hat1, y_hat2 = self.model(x)
        loss1 = self.criterion1(y_hat1, y1)
        loss2 = self.criterion2(y_hat2, y2)
        loss = loss1 + loss2
        self.log('train_loss', loss)

        for metric_name, metric in self.train_metrics.items():
            if "surface" in metric_name:
                metric(y_hat1.cpu(), y1.cpu())
                self.log(metric_name, metric, on_epoch=True)
            elif "smoothness" in metric_name:
                metric(y_hat2.cpu(), y2.cpu())
                self.log(metric_name, metric, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y1 = batch["surface"]
        y2 = batch["smoothness"]

        y_hat1, y_hat2 = self.model(x)
        loss1 = self.criterion1(y_hat1, y1)
        loss2 = self.criterion2(y_hat2, y2)
        loss = loss1 + loss2
        self.log('val_loss', loss)

        for metric_name, metric in self.val_metrics.items():
            if "surface" in metric_name:
                metric(y_hat1.cpu(), y1.cpu())
                self.log(metric_name, metric, on_epoch=True, prog_bar=True)
            elif "smoothness" in metric_name:
                metric(preds=y_hat2.cpu(), target=y2.cpu())
                self.log(metric_name, metric, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
