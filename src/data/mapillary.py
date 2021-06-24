import json
import logging
import os
import requests
from typing import Tuple
from pathlib import Path
from urllib.request import urlretrieve
from PIL import Image as PILImage

from src.config import MAPILLARY_CLIENT_ID

logger = logging.getLogger(__name__)


def download_mapillary_image_information(url: str, file_path: str = None) -> dict:
    """download information for mapillary data for a given url
    and save these metadata to a given file path (json format)
    - inspired from:
    https://blog.mapillary.com/update/2020/02/06/mapillary-data-in-jupyter.html

    Args:
        url (str): url to request information for
        file_path (str, Optional): path to json file to save informaiton in

    Returns:
        dict: dictionary containing all output data in geo json format
    """
    # create an empty GeoJSON to collect all images we find
    output = {"type": "FeatureCollection", "features": []}
    logger.debug("Request URL: {}".format(url))

    # get the request with no timeout in case API is slow
    r = requests.get(url, timeout=None)

    # check if request failed, if failed, keep trying
    while r.status_code != 200:
        r = requests.get(url, timeout=200)
        logger.error("Request failed with URL {}".format(url))

    # retrieve data and save them to output dict
    data = r.json()
    data_length = len(data['features'])
    for feature in data['features']:
        output['features'].append(feature)

    if data_length == 0:
        logger.warning("No data available for the request")

    # if we receive 500 items, there should be a next page
    while data_length == 500 and 'next' in r.links.keys():

        # get the URL for a next page
        link = r.links['next']['url']

        # retrieve the next page in JSON format
        r = requests.get(link)

        # try again if the request fails
        while r.status_code != 200:
            r = requests.get(url, timeout=200)

        # retrieve data and save them to output dict
        data = r.json()
        for feature in data['features']:
            output['features'].append(feature)

        data_length = len(data['features'])  # update data length
        logger.debug('Total images: {}'.format(len(output['features'])))

    # send collected features to the local file
    if file_path is not None:
        with open(file_path, 'w') as outfile:
            json.dump(output, outfile)

    logger.debug('Total images: {}'.format(len(output['features'])))

    return output


def download_mapillary_image_information_by_bbox(bbox: Tuple[float], min_quality_score: int = 4) \
        -> dict:
    """Downloads Mapillary image information of all images that are within a specified bounding box

    Args:
        bbox: Specified bounding box
        min_quality_score: Minimum quality score of the images (1-5, 1 is worst 5 is best)

    Returns:
        Dictionary which contains all the information as specified in the example response here
        https://www.mapillary.com/developer/api-documentation/#search-images
    """
    # Filter by a bounding box on the map, given as min_longitude,min_latitude,max_longitude,
    # max_latitude (lower left, upper right)
    bbox_str = ",".join(map(str, bbox))

    # sort_by=key enables pagination
    url = (
        'https://a.mapillary.com/v3/images?client_id={}&bbox={}&per_page=500&sort_by=key&min_quality_score={}'
    ).format(MAPILLARY_CLIENT_ID, bbox_str, min_quality_score)

    # download data from given URL
    return download_mapillary_image_information(url)


def download_mapillary_image_by_key(image_key: str, download_dir: str):
    """Downloads Mapillary image

    Args:
        image_key: Key of the image that should be downloaded
        download_dir: Directory in which the image is saved as {image_key}.jpg

    Returns:
        None
    """
    image_local_path = os.path.join(download_dir, "{}.jpg".format(image_key))
    if not os.path.isfile(image_local_path):
        url = "https://images.mapillary.com/{}/thumb-2048.jpg".format(image_key)
        urlretrieve(url, image_local_path)
    else:
        logger.info(f"{image_local_path} already exists. Skipping Download.")


def download_mapillary_object_detection_by_key(image_key: str, download_dir: str):
    """Downloads Mapillary object detection

    Args:
        image_key: Key of the image which object detection should be downloaded
        download_dir: Directory in which the object detection json is saved as {image_key}.json

    Returns:
        None
    """
    json_local_path = os.path.join(download_dir, "{}.json".format(image_key))
    if not os.path.isfile(json_local_path):
        layer = "segmentations"
        # request for object detection layer of a certain image (given by image key)
        url = (
            "https://a.mapillary.com/v3/images/{}/object_detections/{}?client_id={}"
        ).format(image_key, layer, MAPILLARY_CLIENT_ID)
        r = requests.get(url, timeout=300)
        data = r.json()

        with open(json_local_path, 'w') as f:
            json.dump(data, f)
    else:
        logger.info(f"{json_local_path} already exists. Skipping Download.")


def crop_image_flat(img_file: str, obj_detections: dict, output_folder: str) -> None:
    """
    Crops a given image to an axis-parallel rectangle which contains all Mapillary object 
    dectections of a 'flat' type

    Args:
        img_file (str): path to image downloaded from mapillary
        obj_dectections (dict): json containing object detections as downloaded 
            from mapillary
        output_folder (str): folder to save cropped image in

    Returns:
        None  
    """
    # Get min and max x and y coordinates for all flat segments associated with the given image
    segments_flat = [(d['properties']['image_key'], 
                 d['properties']['value'], 
                 d['properties']['shape']['coordinates'], 
                 d['properties']['score']) 
                for d in obj_detections['features'] if d['properties']['value'].startswith('construction--flat--')]


    # in case there are any relevant properties, save image, if not do nothing
    if bool(segments_flat):
    
        # get all points of all flat segments
        coords = [point for seg in segments_flat for point in seg[2][0]]
        x_coords = [point[0] for point in coords]
        y_coords = [point[1] for point in coords]
        # careful: these are scaled between 0 and 1
        x_min, x_max, y_min, y_max = min(x_coords), max(x_coords), min(y_coords), max(y_coords) 
    
        # Open and crop the image
        img = PILImage.open(img_file)
        width, height = img.size
        cropped_image = img.crop((x_min * width, y_min * height, x_max * width, y_max * height))
    
        # Create output filename and save cropped image under same name as in input folder
        out_file = Path(img_file).stem
        cropped_image.save(f'{output_folder}/{out_file}.jpg')