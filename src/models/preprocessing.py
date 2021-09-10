import math
from typing import Callable

import torch
import torchvision
from torchvision import transforms


class CustomCropToLowerXPercent(torch.nn.Module):
    """
    crops an image to the bottom x percent of an image;
    can be integrated to a pytorch.transform.Compose pipeline
    """

    def __init__(self, height_perc: int):
        super().__init__()
        self.height_perc = height_perc

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        channels, height, width = img.size()
        new_height = math.floor(height * self.height_perc / 100)
        cropped = torchvision.transforms.functional.crop(
            img, top=height - new_height, left=0, height=new_height, width=width
        )
        return cropped


CROP_TO_LOWER_PERCENTAGE = 50
OUTPUT_SIZE = (256, 256)


def get_train_image_transform() -> Callable:
    return transforms.Compose(
        [
            CustomCropToLowerXPercent(CROP_TO_LOWER_PERCENTAGE),
            transforms.Resize(700),  # get an image with height 700px
            transforms.CenterCrop((700, 1000)),  # get the center part with a HxW of 700x1000
            transforms.RandomCrop(500),  # get a 500x500px square
            transforms.Resize(OUTPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_predict_image_transform() -> Callable:
    return transforms.Compose(
        [
            CustomCropToLowerXPercent(CROP_TO_LOWER_PERCENTAGE),
            transforms.Resize(700),  # get an image with height 700px
            transforms.CenterCrop((700, 1000)),  # get the center part with a HxW of 700x1000
            transforms.RandomCrop(500),  # get a 500x500px square; TODO: Keep it like that?
            transforms.Resize(OUTPUT_SIZE),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
