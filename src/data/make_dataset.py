# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pyrosm

from osm import get_city_geometry, get_mapillary_keys_from_network


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('city', type=click.STRING)
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, city, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    region_osm = pyrosm.OSM(input_filepath)
    city_geometry = get_city_geometry(region_osm, city)

    city_osm = pyrosm.OSM(input_filepath, bounding_box=city_geometry)
    driving_network = city_osm.get_network(
        network_type="driving",
        extra_attributes=["surface", "smoothness"]
    )
    # TODO: get also cycling lanes
    # TODO: check all possible options

    street_mapillary_dict = get_mapillary_keys_from_network(driving_network)

    # TODO: get every image and save it somewhere
    # TODO: make dataset in which we save image name and surface and smoothness
    # TODO: save image separately or directly in binary dataset file?


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
