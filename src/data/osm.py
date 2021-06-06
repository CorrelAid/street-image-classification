from typing import Optional

import geopandas
import shapely
from shapely.geometry import Point, MultiLineString

from src.data.mapillary import download_mapillary_image_information_by_bbox


def add_mapillary_key_to_network(network: geopandas.GeoDataFrame,
                                 street_buffer: float = 1,
                                 shorten_street_by: float = 0,
                                 min_quality_score: int = 4) -> geopandas.GeoDataFrame:
    """Get mapillary keys for all images for each street in the network.

    Args:
        network (geopandas.GeoDataFrame): Street network of edges loaded from OSM with Pyrosm
        street_buffer (float, Optional): Specifies the buffer in meters within that all images
            around each street are considered
        shorten_street_by (float, Optional): Specifies, how many meters of the street are
            excluded at each end. This avoids selecting images belonging to the next street.
            The optimal value depends on the value of street_buffer and for each street on the
            angle at which they intersect.
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

        # to avoid including images from crossing belonging to the wrong road, the ends of each
        # road are excluded by cutting X meters from the LineString
        for idx in range(len(gs)):
            # merge a MultiLineString to a LineString, as substring() only supports LineStrings
            line = shapely.ops.linemerge(gs.iloc[idx])
            # check that
            if not isinstance(line, shapely.geometry.LineString):
                raise NotImplementedError("Only LineStrings are supported, not ", type(line))
            if line.length < 2 * shorten_street_by:
                # skip the street, if it is to short
                continue
            shortened_street = shapely.ops.substring(line,
                                              start_dist=shorten_street_by,
                                              end_dist=line.length - shorten_street_by)
            gs.iloc[idx] = shortened_street

        # cap_style=2 creates a flat ending of the buffer, not an round one
        gs = gs.buffer(street_buffer, cap_style=2)
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
                new_street["img_coords"] = point
                new_df = new_df.append(new_street)

    return new_df

