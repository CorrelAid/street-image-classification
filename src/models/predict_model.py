import flask
import torch
import torchvision
from typing import Tuple
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from urllib.request import urlretrieve
import shutil
import json

from src.config import MAPILLARY_TOKEN_v4
from src.models.model import CargoRocketModel
from src.models.dataset import StreetImageDataset
from src.models.preprocessing import get_predict_image_transform

from src.config import PROJECT_ROOT_PATH
import requests
import traceback

from flask import Flask, jsonify, request

load_dotenv(find_dotenv())

app = Flask(__name__)


# Load model so that it is always available
checkpoint_path = (
    PROJECT_ROOT_PATH
    / "models"
    / "mobilenetv3_large-epoch=97-val_loss=0.93-accuracy_val_surface=0.9231.ckpt"
)

MODEL = CargoRocketModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
MODEL.eval()  # set model to evaluation mode


def download_mapillary_image(key: str, tmp_dir: Path) -> Path:
    """Downloads a certain Mapillary image (given by the key)
    to a temporary directory

    Args:
        key (str): Mapillary image key
        tmp_dir (Path): Temporary folder to save the downloaded
            images to

    Returns:
        Path: path to saved image
    """
    img_name = f"{key}.jpeg"
    local_path = tmp_dir / img_name

    url = f"https://graph.mapillary.com/{key}?access_token={MAPILLARY_TOKEN_v4}&fields=id,thumb_2048_url"
    r_image = requests.get(url, timeout=300)

    fields_dict = r_image.json()
    image_url = fields_dict["thumb_2048_url"]
    urlretrieve(image_url, local_path)

    return local_path


def transform_image(img_path: Path) -> torch.Tensor:
    """Transforms an image in the same way as the training images

    Args:
        img_path (Path): path to image
    """
    # define transformation steps
    crop_to_lower_percentage = 50
    output_size = (256, 256)
    transform = get_predict_image_transform()

    # load image
    img = torchvision.io.read_image(str(img_path)).float()

    # perform transformation
    input = transform(img)

    # single image instead of batch
    input = input.unsqueeze(0)

    return input


def predict_image(img_tensor: torch.Tensor) -> Tuple[str, str]:
    """Predicts a given image tensor and returns the predicted
    smoothness and surface strings

    Args:
        img_tensor (torch.Tensor): Image to predict

    Returns:
        Tuple[str, str]: surface, smoothness as string representation
    """
    surface_pred, smoothness_pred = MODEL(img_tensor)

    surface_id = surface_pred.argmax(axis=1).item()
    smoothness_id = smoothness_pred.argmax(axis=1).item()

    surface_string = StreetImageDataset.get_surface_by_id(surface_id)
    smoothness_string = StreetImageDataset.get_smoothness_by_id(smoothness_id)

    return smoothness_string, surface_string


@app.route("/predict", methods=["GET"])
def predict() -> flask.Response:
    """Takes in one or several mapillary image keys using
    the get field "mapillary_keys" which includes either a
    string of one image or a list with several images

    Returns:
        str: prediction in the form {image_id: {smoothness: xyz,
        surface: xyz}}
    """
    try:
        # read mapillary keys
        mapillary_keys = request.args["mapillary_keys"]
        image_ids = json.loads(mapillary_keys)

        # put single image into a list
        if isinstance(image_ids, str):
            image_ids = [image_ids]

        # create temporary directory
        tmp_dir = Path(__file__).parents[0] / "tmp_dir"
        Path(tmp_dir).mkdir(exist_ok=True)

        # iterate over all keys and get a prediction
        predictions = {}
        for image_id in image_ids:
            print("Getting prediction for image_id={image_id}".format(image_id=image_id))
            image_save_path = download_mapillary_image(image_id, tmp_dir)
            img_tensor = transform_image(image_save_path)
            smoothness, surface = predict_image(img_tensor)
            predictions[image_id] = {"smoothness": smoothness, "surface": surface}

        # delete temporary directory
        shutil.rmtree(tmp_dir)
        return jsonify(predictions)
    except:
        # in case of any error, return the traceback
        return jsonify({"trace": traceback.format_exc()})


if __name__ == "__main__":

    app.run(debug=True)
