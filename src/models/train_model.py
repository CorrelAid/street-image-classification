import torch
from torchvision import transforms
import pytorch_lightning as pl

from src.config import PROJECT_ROOT_PATH
from src.models.dataset import StreetImageDataset, create_train_val_loader
from src.models.model import StreetImageModel
from src.models.preprocessing import CustomCropToLowerXPercent


if __name__ == "__main__":
    # parameters
    dataset_path = "/home/snickels/Projects/street-image-classification/data/processed/dataset_v2"
    crop_to_lower_percentage = 33
    train_ratio = 0.8
    batch_size = 64
    num_workers = 4
    learning_rate = 0.05

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
    num_classes = len(dataset.get_classes())

    print(f"Loaded dataset with {len(dataset)} datapoints and {num_classes} classes.")

    # split
    train_loader, val_loader = create_train_val_loader(
        dataset,
        train_ratio=train_ratio,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # training logic
    model = StreetImageModel(
        num_classes=num_classes,
        learning_rate=learning_rate
    )
    trainer = pl.Trainer(
        default_root_dir=PROJECT_ROOT_PATH,
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=num_epochs
    )
    trainer.fit(model, train_loader, val_loader)
