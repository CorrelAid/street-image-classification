import math

import torch
import torchvision


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
        cropped = torchvision.transforms.functional.crop(img,
                                                         top=height - new_height,
                                                         left=0,
                                                         height=new_height,
                                                         width=width)
        return cropped
