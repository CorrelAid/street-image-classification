import torch

from src.models.model import StreetImageModel
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from urllib.request import urlretrieve

from src.config import MAPILLARY_TOKEN
from src.models.preprocessing import CustomCropToLowerXPercent
from torchvision import transforms
import torchvision
from PIL import Image
from src.config import PROJECT_ROOT_PATH


def download_mapillary_image(key: str) -> Path:
    """Downloads a certain Mapillary image (given by the key)
    to a temporary directory

    Args:
        key (str): Mapillary image key

    Returns:
        Path: path to saved image
    """
    # create temporary directory 
    tmp_dir = Path(__file__).parents[0] / "tmp_dir"
    Path(tmp_dir).mkdir(exist_ok=True)
    img_name = f"{key}.jpeg"
    local_path = tmp_dir / img_name

    url = f"https://graph.mapillary.com/{image_id}?access_token={MAPILLARY_TOKEN}"
    urlretrieve(url, local_path)

    return local_path


def transform_image(img_path: Path) -> None:
    """Transforms an image in the same way as the training images

    Args:
        img_path (Path): path to image
    """
    # define transformation steps
    crop_to_lower_percentage = 33
    resize_size = (500, 500)
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        CustomCropToLowerXPercent(crop_to_lower_percentage),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load image
    img = torchvision.io.read_image(str(img_path)).float()
    # perform transformation
    input = transform(img)

    # unsqueeze batch dimension, in case you are dealing with a single image
    #input = input.unsquueeze(0)
    return input


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    image_id = "469509627711314"
    image_save_path = download_mapillary_image(image_id)
    img_tensor = transform_image(image_save_path)

    checkpoint_path = PROJECT_ROOT_PATH / "models" / "mobilenetv3_large-epoch=97-val_loss=0.93-accuracy_val_surface=0.9231.ckpt"

    # Get model
    model = StreetImageModel.load_from_checkpoint(checkpoint_path=checkpoint_path)

    # Predict
    model.eval()
    model.freeze()
    outputs = model(img_tensor)
    _, preds = torch.max(outputs, 1)

    #smoothness_to_id = {"bad": 2, "intermediate":1, "good": 0}
    #surface_to_id = {"cobblestone": 2, "unpaved": 1, "paved": 0}

    print(f"Prediction for Mapillary image key {image_id} is {preds[0]}")
