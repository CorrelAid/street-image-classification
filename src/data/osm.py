from typing import Optional

import geopandas
from shapely.geometry import Point

from mapillary import download_mapillary_image_information_by_bbox


def add_mapillary_key_to_network(network: geopandas.GeoDataFrame, street_buffer: float = 1,
                                 min_quality_score: int = 4) -> geopandas.GeoDataFrame:
    """Get mapillary keys for all images for each street in the network.

    Args:
        network (geopandas.GeoDataFrame): Street network of edges loaded from OSM with Pyrosm
        street_buffer (float, Optional): Specifies the buffer in meters within that all images
            around each street are considered
        min_quality_score (int, Optional): Specifies the minimum Mapillary quality score (1-5)

    Returns:
        geopandas.GeoDataFrame: Network which was passed as an argument with added column
            mapillary_key. If there are multiple images for a street on Mapillary fitting our
            requirements, the street will occur in multiple rows, each row with another
            mapillary_key.
    """

    new_df = geopandas.GeoDataFrame()

    for _, street in network.iterrows():
        # We put the street geometry in a GeoSeries so that we can use GeoSeries.to_crs()
        # We project the street from EPSG 4326 into EPSG 3857 because in that CRS one unit
        # equals one meters, and after buffering the street with buffer size street_buffer we
        # project it back to EPSG 4326
        gs = geopandas.GeoSeries([street.geometry]).set_crs(epsg=4326)
        gs = gs.to_crs(epsg=3857)
        gs = gs.buffer(street_buffer)
        gs = gs.to_crs(epsg=4326)
        street_geometry = gs[0]
        street_bbox = street_geometry.bounds

        mapillary_dict = download_mapillary_image_information_by_bbox(street_bbox,
                                                                      min_quality_score)
        for photo in mapillary_dict["features"]:
            mapillary_key = photo["properties"]["key"]
            photo_geometry = photo["geometry"]

            # Make sure it is a point, because we don't know what other shapes could be returned
            type_ = photo_geometry["type"]
            if type_ != "Point":
                raise Exception(f"Invalid type '{type_}'.")

            coords = photo_geometry["coordinates"]
            point = Point(coords[0], coords[1])

            # If photo is within buffered street, add a new entry to our new dataframe
            if street_geometry.contains(point):
                new_street = street.copy()
                new_street["mapillary_key"] = mapillary_key
                new_df = new_df.append(new_street)

    return new_df

