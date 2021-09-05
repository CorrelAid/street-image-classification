street_image_classification
==============================
Repository for the project on classifying street image data - in cooperation with 
[CargoRocket](https://cargorocket.de/).

Setting up
------------
* Setup virtual environment and install dependencies listed in `requirements.txt`
* Copy `.env.example` to `.env` and set the variables

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │   └── dl-training.ipynb           <- Preprocessing and training of models using the downloaded dataset.
    │   └── image-transforms-test.ipynb <- Visualize some dataset images, test some transformations for the 
    │                                              training and get a statistics of the occuring image sizes of the dataset.
    │   └── Object Detections.ipynb     <- For some images get the object detections and crop the upper part of 
    │                                              the images to only contain street surface and not the sky.  
    │   └── osm_mapillary.ipynb         <- Tests to read a Open Street Map .pbf file of Berlin, extract
    │                                              cycle lanes and get the mapillary image keys for the extracted cycle 
    │                                              lanes/streets. 
    │   └── osm_statistics.ipynb        <- Read a a Open Street Map .pbf file of Berlin and get statistics of
    │                                              the tagged surfaces and smoothnesses.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── download_images.py
    │   │   └── download_object_detections.py
    │   │   └── image_postprocessing.py
    │   │   └── make_dataset.py
    │   │   └── postprocessing_csv.py
    │   │   └── mapillary.py
    │   │   └── osm.py
    │   │   └── sort_data.py

    │   │
    │   ├── models  <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── dataset.py          <- pytorch Dataset class of the image data and labels. 
    │   │   ├── model.py            <- MobileNetV3 multi-task learning model pytorch_lightning module
    │   │   ├── predict_model.py    <- Flask API to get a prediction for a mapillary image key
    │   │   ├── preprocessing.py    <- Specify transformations of the dataset for training and prediction.
    │   │   └── train_model.py      <- Script to train (probably old version) of the model.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
