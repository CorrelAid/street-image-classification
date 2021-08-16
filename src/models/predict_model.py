import torch
import torchvision

from src.models.model import CargoRocketModel
from src.models.dataset import StreetImageDataset
from src.models.preprocessing import get_predict_image_transform


if __name__ == "__main__":
    checkpoint_path = "/home/snickels/Projects/street-image-classification/mobilenetv3_large-epoch=97-val_loss=0.93-accuracy_val_surface=0.9231.ckpt"
    image_path = "/home/snickels/Projects/street-image-classification/data/processed/dataset_v2/images/ZZS0WXfl05i48OVDg0N5sJ.jpg"
    image = torchvision.io.read_image(image_path).float()

    # TODO: Replace usage of image_path with getting an mapillary image to

    # Get model
    transform = get_predict_image_transform()
    model = CargoRocketModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # Predict
    model.eval()
    model.freeze()

    transformed_image = transform(image)
    outputs = model(transformed_image.unsqueeze(0))

    _, surface_preds = torch.max(outputs[0], 1)
    _, smoothness_preds = torch.max(outputs[1], 1)
    surface_pred = surface_preds[0].item()
    smoothness_pred = smoothness_preds[0].item()

    predicted_surface = StreetImageDataset.get_surface_by_id(surface_preds[0].item())
    predicted_smoothness = StreetImageDataset.get_smoothness_by_id(smoothness_preds[0].item())

    print(f"Prediction for image {image_path} is {predicted_surface} and {predicted_smoothness}")
