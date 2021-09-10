import os
from typing import Callable, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision


class StreetImageDataset(Dataset):
    smoothness_to_id = {"bad": 2, "intermediate": 1, "good": 0}
    surface_to_id = {"cobblestone": 2, "unpaved": 1, "paved": 0}

    """
    Class for street image dataset.
    """

    def __init__(self, path: str, transform: Optional[Callable] = None):
        self._init_csv_df(path)

        self.images_path = f"{path}/images"
        self.transform = transform

    def _init_csv_df(self, path: str):
        self.csv_df = pd.read_csv(f"{path}/data.csv")

    @classmethod
    def get_smoothness_by_id(cls, identifier: int) -> Optional[str]:
        for key, value in cls.smoothness_to_id.items():
            if value == identifier:
                return key

    @classmethod
    def get_surface_by_id(cls, identifier: int) -> Optional[str]:
        for key, value in cls.surface_to_id.items():
            if value == identifier:
                return key

    def get_smoothness_classes(self) -> List[str]:
        return list(self.smoothness_to_id.keys())

    def get_surface_classes(self) -> List[str]:
        return list(self.surface_to_id.keys())

    def __len__(self) -> int:
        return self.csv_df.shape[0]

    def __getitem__(self, idx) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.csv_df.iloc[idx]

        image_path = os.path.join(self.images_path, f"{row['mapillary_key']}.jpg")

        try:
            image = torchvision.io.read_image(image_path).float()
        except RuntimeError as e:
            print("Exception:", e)
            print("image:", image_path)
            raise e

        smoothness_label = self.smoothness_to_id[row["smoothness_category"]]
        surface_label = self.surface_to_id[row["surface_category"]]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "surface": surface_label, "smoothness": smoothness_label}


def split_train_val(dataset: Dataset, train_ratio: float) -> Tuple[Dataset, Dataset]:
    """
    Split Dataset into train and validation dataset.
    """
    if train_ratio < 0.01 or train_ratio > 0.99:
        raise Exception("Train ratio should be between 0.01 and 0.99")

    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset


def create_train_val_loader(
    dataset: Dataset, train_ratio: float, batch_size: int, num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader for train and validation dataset.
    """
    train_dataset, val_dataset = split_train_val(dataset, train_ratio)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader
