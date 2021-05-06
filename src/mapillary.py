import json
import requests
from typing import Tuple

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


def download_mappilary_image_information_by_bbox(bbox: Tuple[float]) -> dict:
    # Filter by a bounding box on the map, given as min_longitude,min_latitude,max_longitude,
    # max_latitude (lower left, upper right)
    bbox_str = ",".join(map(str, bbox))

    # define image qualities to consider (1-5 inclusively)
    min_score = 3
    max_score = 5

    print(MAPILLARY_CLIENT_ID)

    # sort_by=key enables pagination
    url = (
        'https://a.mapillary.com/v3/images?client_id={}&bbox={}&per_page=500&sort_by=key&min_quality_score={}&max_quality_score={}' \
    ).format(MAPILLARY_CLIENT_ID, bbox_str, min_score, max_score)

    # download data from given URL
    return download_mapillary_image_information(url)
