"""This file includes a function to sort the 
dataset according to its labels in separate 
subfolders. This is necessary for being able
to use the Pytorch dataloader
"""
from pathlib import Path
from numpy.lib.type_check import imag 
import pandas as pd 
#import geopandas
import os 
import shutil 

def load_image_metadata(path_to_data_csv: str) -> \
    pd.DataFrame:
    """Loads the image metadata, i.e. tags, mapillary id etc. 

    Args:
        path_to_data_csv (str): path to data.csv file

    Returns:
        pd.DataFrame: dataframe with read data
    """
    return pd.read_csv(path_to_data_csv)


if __name__ == "__main__":
    path_to_data_csv = Path(__file__).parents[2] / \
        "data" / "processed" / "data.csv"
    path_to_images =  Path(__file__).parents[2] / \
        "data" / "processed" / "images"
    path_for_dataset = Path(__file__).parents[2] / \
        "data" / "processed" / "final"

    try:
        os.mkdir(path_for_dataset)
    except:
        pass 

    image_data = load_image_metadata(path_to_data_csv)

    smoothness_tags = image_data["smoothness"].unique()

    # create one folder per tag
    for tag in smoothness_tags:
        try:
            os.mkdir(path_for_dataset / tag)
        except:
            pass

    # copy images according to its tag to the final dataset
    # folder
    for _, img in image_data.iterrows():
        key = img["mapillary_key"]
        smoothness = img["smoothness"]
        
        # path to original image
        img_name = key + ".jpg"
        orig_file = path_to_images / img_name

        # new path
        new_file = path_for_dataset / smoothness / img_name
        shutil.copy(orig_file, new_file)

    