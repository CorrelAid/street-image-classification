import copy
import math
from pathlib import Path
import time
from typing import Callable, Optional, Tuple
import os

import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models


class CustomCropToLowerXPercent:
    """
    crops an image to the bottom x percent of an image;
    can be integrated to a pytorch.transform.Compose pipeline
    """
    def __init__(self, height_perc: int):
        self.height_perc = height_perc

    def __call__(self, img: torch.Tensor):
        channels, height, width = img.size()
        new_height = math.floor(height * self.height_perc / 100)
        cropped = transforms.functional.crop(img,
                                             top=height - new_height, 
                                             left=0, 
                                             height=new_height, 
                                             width=width)
        return cropped


class StreetImageDataset(Dataset):
    """
    Class for street image dataset.
    """
    def __init__(self, path: str, label_column: str, query: Optional[str] = None,
                 transform: Optional[Callable] = None):
        self._init_csv_df(path, query)
        self._init_label_column(label_column)

        self.images_path = f"{path}/images"
        self.transform = transform

    def _init_csv_df(self, path: str, query: Optional[str] = None):
        csv_df = pd.read_csv(f"{path}/data.csv")
        if query:
            csv_df = csv_df.query(query)
        self.csv_df = csv_df

    def _init_label_column(self, label_column: str):
        self.label_column = label_column
        self.label_to_id = {label: i for i, label in enumerate(self.csv_df[label_column].unique())}
        self.id_to_label = {i: label for i, label in enumerate(self.csv_df[label_column].unique())}

    def get_classes(self):
        return

    def __len__(self) -> int:
        return self.csv_df.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.csv_df.iloc[idx]

        image_path = os.path.join(self.images_path, f"{row['mapillary_key']}.jpg")
        image = torchvision.io.read_image(image_path).float()
        label = self.label_to_id[row[self.label_column]]

        if self.transform:
            image = self.transform(image)

        return image, label


def split_train_test(dataset: Dataset, train_ratio: float) -> Tuple[Dataset, Dataset]:
    """
    Split Dataset into train and test dataset.
    """
    if train_ratio < 0.01 or train_ratio > 0.99:
        raise Exception("Train ratio should be between 0.01 and 0.99")

    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    return train_dataset, test_dataset


def create_train_test_loader(dataset: Dataset, train_ratio: float, batch_size: int) \
        -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader for train and test dataset.
    """
    train_dataset, test_dataset = split_train_test(dataset, train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


if __name__ == "__main__":
    # parameters
    dataset_path = "/home/snickels/Projects/street-image-classification/data/processed/dataset_v2"
    crop_to_lower_percentage = 33
    train_ratio = 0.8
    batch_size = 64
    learning_rate = 0.3

    num_epochs = 10

    # transforms
    resize_size = (500, 500)
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        CustomCropToLowerXPercent(crop_to_lower_percentage),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load dataset
    dataset = StreetImageDataset(
        dataset_path,
        label_column="smoothness_category",
        query="surface_category == 'paved'",
        transform=transform
    )

    # split
    train_loader, test_loader = create_train_test_loader(dataset, train_ratio=train_ratio,
                                                         batch_size=batch_size)

    # setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True)

    # freeze the pre-trained layers, so we don’t backprop through them during training
    for param in model.parameters():
        param.requires_grad = False

    # redefine fully connected layer
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 10),
        nn.LogSoftmax(dim=1)
    )

    model.to(device)

    # loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    # training and evaluation
    # TODO: Add progress bar
    # TODO: Add saving/loading after a single epoch
    print("Training started")

    since = time.time()
    best_test_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        model.train()

        running_train_loss = 0.0
        running_train_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            running_train_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_train_loss / len(train_loader)
        epoch_train_acc = float(running_train_corrects) / (len(train_loader) * batch_size)

        print(f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}")

        model.eval()

        running_test_loss = 0.0
        running_test_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item()
                running_test_corrects += torch.sum(preds == labels.data)

        epoch_test_loss = running_test_loss / len(train_loader)
        epoch_test_acc = float(running_test_corrects) / (len(train_loader) * batch_size)

        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc

        print(f"Test Loss: {epoch_test_loss:.4f} Acc: {epoch_test_acc:.4f}")

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Test Acc: {:4f}'.format(best_test_acc))
