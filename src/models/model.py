import torch
import torchmetrics
import torchvision
import pytorch_lightning as pl
from typing import Union, Optional, Callable, List, Tuple, Any
from torch import nn

def get_metric_dict(mode: str = "train", num_classes: int = 3) -> dict:
    kwargs = {"num_classes": num_classes, "average": "weighted"}
    metric_dict = {f"accuracy_{mode}_surface": torchmetrics.Accuracy(**kwargs),
                   f"precision_{mode}_surface": torchmetrics.Precision(**kwargs),
                   f"accuracy_{mode}_smoothness": torchmetrics.Accuracy(**kwargs),
                   f"precision_{mode}_smoothness": torchmetrics.Precision(**kwargs),
                   f"f1_{mode}_surface": torchmetrics.F1(**kwargs),
                   f"f1_{mode}_smoothness": torchmetrics.F1(**kwargs),
                  }
    return metric_dict

class MTL_MobileNetV3(torchvision.models.MobileNetV3):
    def __init__(self, inverted_residual_setting, last_channel):
        super().__init__(inverted_residual_setting, last_channel)
        arch = "mobilenet_v3_large"
        self.inverted_residual_setting = inverted_residual_setting
        self.last_channel = last_channel
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        self.lastconv_output_channels = 6 * lastconv_input_channels

    def add_mtl_head(self, num_classes=3):
        self.classifier =  nn.Sequential(
                nn.Linear(self.lastconv_output_channels, self.last_channel),
                nn.Hardswish(inplace=True),
                nn.Dropout(p=0.2, inplace=True))
        self.classifier1 = nn.Sequential(nn.Linear(self.last_channel, num_classes), 
                                         nn.LogSoftmax(dim=1))
        self.classifier2 = nn.Sequential(nn.Linear(self.last_channel, num_classes), 
                                         nn.LogSoftmax(dim=1))
        
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)
        x1 = self.classifier1(x)
        x2 = self.classifier2(x)
        return x1, x2


def my_mobilenet_v3_model(
    arch: str,
    inverted_residual_setting,
    last_channel: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
):
    model = MTL_MobileNetV3(inverted_residual_setting, last_channel)
    if pretrained:
        if torchvision.models.mobilenetv3.model_urls.get(arch, None) is None:
            raise ValueError("No checkpoint is available for model type {}".format(arch))
        state_dict = torchvision.models.mobilenetv3.load_state_dict_from_url(torchvision.models.mobilenetv3.model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def mtl_mobilenet_v3_large(pretrained: bool = True, num_classes: int = 3):
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = torchvision.models.mobilenetv3._mobilenet_v3_conf(arch)
    m = my_mobilenet_v3_model(arch, inverted_residual_setting, last_channel, pretrained, progress=True)
    m.add_mtl_head(num_classes=3)
    return m


class CargoRocketModel(pl.LightningModule):

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.model = mtl_mobilenet_v3_large(pretrained=True, num_classes=self.num_classes)
        self.criterion1 = nn.NLLLoss()
        self.criterion2 = nn.NLLLoss()
        self.learning_rate = 1e-3
        
        self.train_metrics = get_metric_dict(mode="train", num_classes=self.num_classes)      
        self.val_metrics = get_metric_dict(mode="val", num_classes=self.num_classes)
        print("Using", self.num_classes, "classes")
       
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        prediction = self.model(x)
        return prediction

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer