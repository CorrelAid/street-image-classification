# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

import pandas as pd

from mapillary_image_classification.data.mapillary import download_mapillary_object_detection_by_key


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True))
def main(dataset_path):
    """
    This script loads the object detections for a specific dataset.
    """
    logger = logging.getLogger(__name__)
    logger.info("Downloading mapillary object detections for dataset with existing data.csv")

    if not os.path.exists(dataset_path):
        logger.error(f"{dataset_path} does not exist")
        return

    data_csv_path = f"{dataset_path}/data.csv"
    if not os.path.exists(data_csv_path):
        logger.error(f"{data_csv_path} does not exist")
        return

    data = pd.read_csv(data_csv_path)
    mapillary_keys = data["mapillary_key"].unique()

    # Create sub dirs
    image_dir = f"{dataset_path}/images"
    object_detection_dir = f"{dataset_path}/object_detections"

    Path(image_dir).mkdir(exist_ok=True)
    Path(object_detection_dir).mkdir(exist_ok=True)

    for mapillary_key in tqdm(mapillary_keys, desc="Downloading object detections"):
        download_mapillary_object_detection_by_key(mapillary_key, download_dir=object_detection_dir)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
