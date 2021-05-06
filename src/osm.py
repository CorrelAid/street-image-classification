from typing import Dict, List

import geopandas
import pyrosm
from shapely.geometry import Point, Polygon

from mapillary import download_mappilary_image_information_by_bbox


def get_city_geometry(osm: pyrosm.OSM, city_name: str) -> Polygon:
    """
    Get geometry of city in OSM file as a shapely polygon
    """
    boundaries_df = osm.get_boundaries(name=city_name)
    city_boundaries_df = boundaries_df[boundaries_df["admin_level"] == "6"]
    return city_boundaries_df.iloc[0].geometry


def get_mapillary_keys_from_network(network: geopandas.GeoDataFrame, street_buffer: float = 0.0001) \
        -> Dict[int, List[str]]:
    """
    Get mapillary keys for all images for each street in the network
    street_buffer specifies the buffer within that all images around each street are considered
    Returns a dict in which key is the OSM street id and value is a list of mapillary keys
    """
    street_ids_mapillary_keys_dict = {}

    for _, street in network.iterrows():
        street_geometry = street.geometry.buffer(street_buffer)
        street_bbox = street_geometry.bounds
        mapillary_keys = []

        mapillary_dict = download_mappilary_image_information_by_bbox(street_bbox)
        for photo in mapillary_dict["features"]:
            mapillary_key = photo["properties"]["key"]
            photo_geometry = photo["geometry"]

            type_ = photo_geometry["type"]
            if type_ != "Point":
                raise Exception(f"Invalid type '{type_}'.")

            coords = photo_geometry["coordinates"]
            point = Point(coords[0], coords[1])

            if street_geometry.contains(point):
                mapillary_keys.append(mapillary_key)

        street_ids_mapillary_keys_dict[street.id] = mapillary_keys

    return street_ids_mapillary_keys_dict
