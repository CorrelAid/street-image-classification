"""This file includes a function to sort the dataset according to 
its labels in separate subfolders. This is necessary for being able
to directly use the Pytorch dataloader
"""
from pathlib import Path
import pandas as pd 
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


def copy_images_to_smoothness_labeled_folders(image_data: pd.DataFrame,
path_to_images: str, final_dataset_folder: str, surface_tag: str):
    """Creates folders for each smoothness tag of the given surface tag
    and copies the images of the given folder to these subfolders
    according to their smoothness tags. At the end, there are subfolders
    named with each smoothness tags and they contain all images with the
    respective tag

    Args:
        image_data (pd.DataFrame): image dataframe
        path_to_images (str): Path to images containing the images, named
            in dataframe
        final_dataset_folder (str): Path to the final dataset folder
        surface_tag (str): Surface tag to consider 
    """
    # filter data 
    surface_data = image_data[image_data["surface"] == surface_tag]

    # extract all available smoothness tags
    smoothness_tags = surface_data["smoothness"].unique()

    # create one folder per tag
    for tag in smoothness_tags:
        try:
            os.mkdir(os.path.join(final_dataset_folder, tag))
        except:
            pass

    # copy images according to its tag to the final dataset
    # folder
    for _, img in surface_data.iterrows():
        key = img["mapillary_key"]
        smoothness = img["smoothness"]
        
        # path to original image
        img_name = key + ".jpg"
        orig_file = path_to_images / img_name

        # new path
        new_file = path_for_dataset / smoothness / img_name
        shutil.copy(orig_file, new_file)


if __name__ == "__main__":
    path_to_data_csv = Path(__file__).parents[2] / \
        "data" / "balanced_0" / "data.csv"
    path_to_images =  Path(__file__).parents[2] / \
        "data" / "balanced_0" / "images"
    path_for_dataset = Path(__file__).parents[2] / \
        "data" / "balanced_0" / "final"

    try:
        os.mkdir(path_for_dataset)
    except:
        pass 
    
    image_data = load_image_metadata(path_to_data_csv)

    image_data = image_data.iloc[0:100]

    copy_images_to_smoothness_labeled_folders(image_data, 
        path_to_images, path_for_dataset, "asphalt")


    