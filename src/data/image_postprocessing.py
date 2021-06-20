from pathlib import Path 
import os 
import json 
import pandas as pd 
from src.data.mapillary import crop_image_flat


def image_postprocessing(output_dir: str, image_df: pd.DataFrame):
    """Represents a postprocessing function

    Args:
        output_dir (str): directory with saved dataset (containing
            'images' and 'object_detections' subfolders)
        image_df (pd.DataFrame): dataframe containing all mapillary keys
            in the dataset 
    """
    cropped_dir = f"{output_dir}/cropped"
    Path(cropped_dir).mkdir(exist_ok=True)

    image_dir = output_dir / "images"
    obj_dir = output_dir / "object_detections"

    mapillary_keys = image_df["mapillary_key"].to_list()

    # iterate over all keys in the dataset 
    for map_key in mapillary_keys:
        image = os.path.join(image_dir, map_key + ".jpg")
        obj_det = os.path.join(obj_dir, map_key + ".json")

        with open(obj_det, "r") as f:
            obj_json = json.load(f)

        # crop image with object detections
        crop_image_flat(image, obj_json, output_dir)