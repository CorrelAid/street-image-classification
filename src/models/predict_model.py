import torch
from typing import Union
from src.models.model import CargoRocketModel
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from urllib.request import urlretrieve
import shutil
import json 

from src.config import MAPILLARY_TOKEN
from src.models.preprocessing import CustomCropToLowerXPercent
from torchvision import transforms
import torchvision
from PIL import Image
from src.config import PROJECT_ROOT_PATH
import requests
import traceback

from flask import Flask, jsonify, request
load_dotenv(find_dotenv())

app = Flask(__name__)


# Load model so that it is always available
checkpoint_path = PROJECT_ROOT_PATH / "models" / \
        "mobilenetv3_large-epoch=97-val_loss=0.93-accuracy_val_surface=0.9231.ckpt"

MODEL = CargoRocketModel.load_from_checkpoint(checkpoint_path=checkpoint_path)
MODEL.eval() # set model to evaluation mode


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

    url = f"https://graph.mapillary.com/{key}?access_token={MAPILLARY_TOKEN}&fields=id,thumb_2048_url"
    r_image = requests.get(url, timeout=300)

    fields_dict = r_image.json()
    image_url = fields_dict["thumb_2048_url"]
    urlretrieve(image_url, local_path)

    return local_path


def transform_image(img_path: Path) -> None:
    """Transforms an image in the same way as the training images

    Args:
        img_path (Path): path to image
    """
    # define transformation steps
    crop_to_lower_percentage = 50
    output_size = (256, 256)
    transform = transforms.Compose([
        CustomCropToLowerXPercent(crop_to_lower_percentage),
        transforms.Resize(700),  # get an image with height 700px
        transforms.CenterCrop((700, 1000)),  # get the center part with a HxW of 700x1000
        transforms.RandomCrop(500),  # get a 500x500px square
        transforms.Resize(output_size),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load image
    img = torchvision.io.read_image(str(img_path)).float()
    
    # perform transformation
    input = transform(img)

    # single image instead of batch
    input = input.unsqueeze(0)

    return input

def predict_image(img_tensor) -> Union[str, str]:
    """Predicts a given image tensor and returns the predicted
    smoothness and surface strings

    Args:
        img_tensor ([type]): [description]

    Returns:
        Union[str, str]: surface, smoothness
    """
    # predict
    surface_pred, smoothness_pred = MODEL(img_tensor)

    surface_id = surface_pred.argmax(axis=1).item()
    smoothness_id = smoothness_pred.argmax(axis=1).item()

    # match class ids back to actual string class name
    smoothness_to_name = {2: "bad", 1: "intermediate", 0: "good"}
    surface_to_name = {2: "cobblestone", 1: "unpaved", 0: "paved"}

    smoothness_string = smoothness_to_name[smoothness_id]
    surface_string = surface_to_name[surface_id]

    return smoothness_string, surface_string


@app.route('/predict', methods=['GET'])
def predict() -> str:
    """Takes in one or several mapillary image keys using
    the get field "mapillary_keys" which includes either a 
    string of one image or a list with several images

    Returns:
        str: prediction in the form {image_id: {smoothness: xyz, 
        surface: xyz}}
    """
    try:
        # read mapillary keys
        mapillary_keys = request.args['mapillary_keys']
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
            print(image_id)
            image_save_path = download_mapillary_image(image_id, tmp_dir)
            img_tensor = transform_image(image_save_path)
            smoothness, surface = predict_image(img_tensor)
            predictions[image_id] = {"smoothness": smoothness, \
                "surface": surface}

        # delete temporary directory
        shutil.rmtree(tmp_dir)
        return jsonify(predictions)
    except:
        # in case of any error, return the traceback
        return jsonify({'trace': traceback.format_exc()})


if __name__ == "__main__":

    app.run(debug=True)
    
    




    

    
