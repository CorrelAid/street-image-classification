import torch

from src.models.model import StreetImageModel


if __name__ == "__main__":
    checkpoint_path = "xx"
    input_mapillary_key = "xx"

    # TODO: Get image
    input_image = None

    # Get model
    model = StreetImageModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # Predict
    model.eval()
    model.freeze()
    outputs = model(input_image)
    _, preds = torch.max(outputs, 1)

    print(f"Prediction for Mapillary image key {input_mapillary_key} is {preds[0]}")
