import json
import os
import requests
from typing import Tuple
from urllib.request import urlretrieve

from config import MAPILLARY_CLIENT_ID


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
    print("Request URL: {}".format(url))

    # get the request with no timeout in case API is slow
    r = requests.get(url, timeout=None)

    # check if request failed, if failed, keep trying
    while r.status_code != 200:
        r = requests.get(url, timeout=200)
        print("Request failed")

    # retrieve data and save them to output dict
    data = r.json()
    data_length = len(data['features'])
    for feature in data['features']:
        output['features'].append(feature)

    if data_length == 0:
        print("No data available for the request")

    # if we receive 500 items, there should be a next page
    while data_length == 500:

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
        print('Total images: {}'.format(len(output['features'])))

    # send collected features to the local file
    if file_path is not None:
        with open(file_path, 'w') as outfile:
            json.dump(output, outfile)

    print('DONE')  # once all images are saved in a GeoJSON and saved, we finish
    print('Total images: {}'.format(len(output['features'])))

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
    # todo: do we actually want to have pagination?
    url = (
        'https://a.mapillary.com/v3/images?client_id={}&bbox={}&per_page=500&sort_by=key&min_quality_score={}' \
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
    url = "https://images.mapillary.com/{}/thumb-2048.jpg".format(image_key)
    image_local_path = os.path.join(download_dir, "{}.jpg".format(image_key))
    urlretrieve(url, image_local_path)


def download_mapillary_object_detection_by_key(image_key: str, download_dir: str):
    """Downloads Mapillary object detection

    Args:
        image_key: Key of the image which object detection should be downloaded
        download_dir: Directory in which the object detection json is saved as {image_key}.json

    Returns:
        None
    """
    layer = "segmentations"

    # request for object detection layer of a certain image (given by image key)
    url = (
        "https://a.mapillary.com/v3/images/{}/object_detections/{}?client_id={}" \
    ).format(image_key, layer, MAPILLARY_CLIENT_ID)
    r = requests.get(url, timeout=300)
    data = r.json()

    with open(os.path.join(download_dir, "{}.json".format(image_key)), 'w') as f:
        json.dump(data, f)
