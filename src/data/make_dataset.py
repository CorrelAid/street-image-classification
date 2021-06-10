# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import pyrosm

from data.mapillary import download_mapillary_image_by_key, download_mapillary_object_detection_by_key
from data.osm import add_mapillary_key_to_network


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('street_buffer', type=click.FLOAT)
@click.argument('shorten_street_by', type=click.FLOAT)
@click.argument('min_quality_score', type=click.INT)
@click.argument('output_dir', type=click.Path())
def main(input_filepath, street_buffer, shorten_street_by, min_quality_score, output_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    osm = pyrosm.OSM(input_filepath)
    network = osm.get_network(
        network_type="cycling",
        extra_attributes=["surface", "smoothness"]
    )

    # Filter relevant columns
    network = network[["id", "geometry", "surface", "smoothness"]]

    # Filter only records where both surface and smoothness is set
    network = network[(~network["surface"].isna()) & (~network["smoothness"].isna())]

    # Get Mapillary keys for each street
    logger.info("Get mapillary data for street network..")

    street_mapillary_df = add_mapillary_key_to_network(network,
                                                       street_buffer=street_buffer,
                                                       shorten_street_by=shorten_street_by,
                                                       min_quality_score=min_quality_score)

    # Create output dir
    Path(f"{output_dir}").mkdir(parents=True, exist_ok=True)

    # Create sub dirs
    image_dir = f"{output_dir}/images"
    object_detection_dir = f"{output_dir}/object_detections"

    Path(image_dir).mkdir(exist_ok=True)
    Path(object_detection_dir).mkdir(exist_ok=True)

    # Export parameters
    logger.info("Exporting parameters.csv..")

    pd.DataFrame([
        ["street_buffer", street_buffer],
        ["shorten_street_by", shorten_street_by],
        ["min_quality_score", min_quality_score]
    ], columns=["parameter", "value"]).to_csv(f"{output_dir}/parameters.csv", index=False)

    # Export street_mapillary_df as data.csv
    logger.info("Exporting data.csv..")

    street_mapillary_df.to_csv(f"{output_dir}/data.csv", index=False)

    # Download images and object detections
    logger.info("Downloading images and object detections..")

    for _, row in street_mapillary_df.iterrows():
        download_mapillary_image_by_key(row["mapillary_key"], download_dir=image_dir)
        download_mapillary_object_detection_by_key(row["mapillary_key"],
                                                   download_dir=object_detection_dir)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
