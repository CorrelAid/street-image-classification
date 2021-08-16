from pathlib import Path
import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

PROJECT_ROOT_PATH = Path(__file__).parent.parent
MAPILLARY_TOKEN = os.getenv("MAPILLARY_TOKEN")
