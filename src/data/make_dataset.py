# -*- coding: utf-8 -*-
import click
import logging
import os
from typing import Iterator
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import geopandas
import pandas as pd
import pyrosm

from src.data.osm import add_mapillary_key_to_network, define_categories
from src.data.mapillary import (download_mapillary_image_by_key,
                                download_mapillary_object_detection_by_key)


def split_dataframe(df: geopandas.GeoDataFrame, chunk_size: int) \
        -> Iterator[geopandas.GeoDataFrame]:
    num_chunks = len(df) // chunk_size + 1
    for i in range(num_chunks):
        yield df[i * chunk_size:(i + 1) * chunk_size]


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('street_buffer', type=click.FLOAT)
@click.argument('shorten_street_by', type=click.FLOAT)
@click.argument('min_quality_score', type=click.INT)
@click.argument('chunk_size', type=click.INT)
@click.argument('output_dir', type=click.Path())
def main(input_filepath: str,
         street_buffer: float,
         shorten_street_by: float,
         min_quality_score: int,
         chunk_size: int,
         output_dir: str) -> None:
    """Runs data processing scripts to turn raw data to cleaned data for analysis and training.

    The raw data (usually under '../raw') is loaded, cleaned and transformed for analysis and
    training with it and saved (usually under '../processed')

    Args:
        input_filepath: Path to a open street map data file to load the road network from.
        street_buffer: Size of the buffer to draw around each street to search images in.
        shorten_street_by: Length of each end of a street it is shortened by. This avoids
            (some) misclassified images, when a image at an intersection is added as image of
            two streets (with potentially different surfaces and smoothnesses.
        min_quality_score: Minimum quality of the image as categorized by Mapillary. [0...5].
            Usually 4 or 5.
        chunk_size: ?
        output_dir: Directory, where the dataset should be created.

    Returns:
        None

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
    network_size = network.shape[0]

    # Assign values of surface and smoothness to categories
    network = define_categories(network)

    # Get Mapillary keys for each street
    logger.info("Get mapillary data for street network..")

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
        ["min_quality_score", min_quality_score],
        ["chunk_size", chunk_size]
    ], columns=["parameter", "value"]).to_csv(f"{output_dir}/parameters.csv", index=False)

    # Combine Mapillary with OSM data
    data_output_path = f"{output_dir}/data.csv"
    osm_id_output_path = f"{output_dir}/osm_ids.csv"

    street_mapillary_df = None
    streets_processed = 0

    if os.path.exists(data_output_path) and os.path.exists(osm_id_output_path):
        logger.info(f"{data_output_path} already exists, filtering OSM data..")
        # The capitalized arguments fix a weird recursion error that seems to happen whenever there
        # are two coordinates in the csv file (in our case geometry [osm street coords] and
        # mapillary_coordinates [mapillary photo coords])
        street_mapillary_df = geopandas.read_file(
            data_output_path,
            GEOM_POSSIBLE_NAMES="geometry",
            KEEP_GEOM_COLUMNS="NO"
        ).set_crs(epsg=4326)
        osm_id_df = pd.read_csv(osm_id_output_path)

        # Convert id to int
        osm_id_df["id"] = pd.to_numeric(osm_id_df["id"], downcast='integer')
        street_mapillary_df["id"] = pd.to_numeric(street_mapillary_df["id"], downcast='integer')

        # Filter streets that were already used out of the network
        network = network[~network["id"].isin(osm_id_df["id"])]
        network_size_after = network.shape[0]

        streets_processed = network_size - network_size_after

        logger.info(f"OSM data got shrinked from {network_size} to {network_size_after}")

    logger.info("Combining Mapillary with OSM data..")

    for network_partition in split_dataframe(network, chunk_size):
        cur_street_mapillary_df = add_mapillary_key_to_network(network_partition,
                                                               street_buffer=street_buffer,
                                                               shorten_street_by=shorten_street_by,
                                                               min_quality_score=min_quality_score)

        # Save processed OSM ids in a separate file so that we can track after a restart which
        # OSM ids we already processed (if we just took the data.csv we would redo all OSM streets
        # that we could not assign any Mapillary key to)
        network_partition["id"].to_csv(osm_id_output_path,
                                       mode="a",
                                       header=not os.path.exists(osm_id_output_path),
                                       index=False)

        if cur_street_mapillary_df.shape[0] > 0:
            cur_street_mapillary_df["id"] = pd.to_numeric(cur_street_mapillary_df["id"],
                                                          downcast="integer")

            cur_street_mapillary_df.to_csv(data_output_path,
                                           mode="a",
                                           header=not os.path.exists(data_output_path),
                                           index=False)

            if street_mapillary_df is None:
                street_mapillary_df = cur_street_mapillary_df
            else:
                street_mapillary_df = pd.concat([street_mapillary_df, cur_street_mapillary_df])

        streets_processed += network_partition.shape[0]
        logger.info(f"Progress {streets_processed / network_size * 100:.2f}%")

    # Download images and object detections
    logger.info("Downloading images and object detections..")

    for _, row in street_mapillary_df.iterrows():
        download_mapillary_image_by_key(row["mapillary_key"], download_dir=image_dir)
        download_mapillary_object_detection_by_key(row["mapillary_key"],
                                                   download_dir=object_detection_dir)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
