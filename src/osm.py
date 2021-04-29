import geopandas
import pyrosm
from shapely.geometry import Point, Polygon

from mapillary import download_mappilary_image_information_by_bbox


def get_city_geometry(osm: pyrosm.OSM, city_name: str) -> Polygon:
    boundaries_df = osm.get_boundaries(name=city_name)
    city_boundaries_df = boundaries_df[boundaries_df["admin_level"] == "6"]
    return city_boundaries_df.iloc[0].geometry


def get_mappilary_image_ids(network: geopandas.GeoDataFrame):
    for _, street in network.iterrows():
        # TODO: Add buffer around geometry
        street_geometry = street.geometry
        street_bbox = street_geometry.bounds

        mappilary_dict = download_mappilary_image_information_by_bbox(street_bbox)
        for photo in mappilary_dict["features"]:
            photo_geometry = photo["geometry"]

            type_ = photo_geometry["type"]
            if type_ not in ("Point", "LineString"):
                raise Exception("Invalid type")

            coords = photo_geometry["coordinates"]

            if isinstance(coords[0], list):
                # LineString
                for coord in coords:
                    point = Point(coord[0], coord[1])

                    # TODO: Check if point is in street_geometry, if yes, add to image_ids

            elif isinstance(coords[0], float):
                # Point
                point = Point(coords[0], coords[1])

                # TODO: Check if point is in street_geometry, if yes, add to image_ids

        # TODO: Temp
        break