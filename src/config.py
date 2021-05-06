import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

MAPILLARY_CLIENT_ID = os.getenv("MAPILLARY_CLIENT_ID")
