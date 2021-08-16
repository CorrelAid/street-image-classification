import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from src.config import PROJECT_ROOT_PATH
from src.models.dataset import StreetImageDataset, create_train_val_loader
from src.models.model import CargoRocketModel
from src.models.preprocessing import get_train_image_transform


def get_checkpoint_callback(metric: str) -> ModelCheckpoint:
    return ModelCheckpoint(monitor=metric,
                           mode="max",
                           filename='{epoch}-{val_loss:.2f}-{' + metric + ':.4f}')


if __name__ == "__main__":
    # parameters
    dataset_path = "/home/snickels/Projects/street-image-classification/data/processed/dataset_v2"
    train_ratio = 0.7
    batch_size = 64
    num_workers = 4
    learning_rate = 1e-3

    num_epochs = 100

    # load dataset
    dataset = StreetImageDataset(
        dataset_path,
        transform=get_train_image_transform()
    )
    num_surface_classes = len(dataset.get_surface_classes())
    num_smoothness_classes = len(dataset.get_smoothness_classes())

    print(f"Loaded dataset with {len(dataset)} datapoints, {num_surface_classes} surface classes "
          f"and {num_smoothness_classes} smoothness classes.")

    # Hotfix because model only allows one num_class parameter for both classes
    assert num_surface_classes == num_smoothness_classes
    num_classes = num_surface_classes

    # split
    train_loader, val_loader = create_train_val_loader(
        dataset,
        train_ratio=train_ratio,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # training logic
    model = CargoRocketModel(
        num_classes=num_classes,
        learning_rate=learning_rate
    )
    trainer = pl.Trainer(
        default_root_dir=PROJECT_ROOT_PATH,
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=num_epochs,
        callbacks=[
            ModelCheckpoint(save_last=True),
            get_checkpoint_callback(metric="f1_val_surface"),
            get_checkpoint_callback(metric="f1_val_smoothness"),
            get_checkpoint_callback(metric="accuracy_val_surface"),
            get_checkpoint_callback(metric="accuracy_val_smoothness"),
        ]
    )
    trainer.fit(model, train_loader, val_loader)
