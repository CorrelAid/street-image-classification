from pathlib import Path
import os
import json
import pandas as pd
from mapillary_image_classification.data.mapillary import crop_image_flat


def crop_relevant_image_parts(output_dir: str, image_df: pd.DataFrame, 
copy_img_without_obj: bool = False):
    """Iterates over all images in the dataset and takes their
    object detections to crop the image to only contain the relevant parts,
    e.g. streets. The images are saved in a folder "cropped" in the given
    dataset directory

    Args:
        output_dir (str): directory with saved dataset (containing
            'images' and 'object_detections' subfolders)
        image_df (pd.DataFrame): dataframe containing all mapillary keys
            in the dataset
        copy_img_without_obj (bool): if this flag is set to True, images 
            without object detections are copied to the output directory
            as well. Defaults to False
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
        crop_image_flat(image, obj_json, cropped_dir, copy_img_without_obj)


if __name__ == "__main__":
    path_for_dataset = Path(__file__).parents[2] / "data" / "example_dataset"
    path_to_data_csv = Path(__file__).parents[2] / "data" / "example_dataset" / "data.csv"
    data = pd.read_csv(path_to_data_csv)
    crop_relevant_image_parts(path_for_dataset, data, True)