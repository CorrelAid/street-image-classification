Classification of road quality using crowd-sourced images
==========================================================
This is the CorrelAid repository for the project on classifying street image data - in cooperation with 
[CargoRocket](https://cargorocket.de/).

With the goal of promoting sustainable mobility, the volunteer organization CargoRocket is  developing a freely available route planning app for cargo bikes. Because a cargo bike has its own unique road quality requirements, these must be taken into account when planning a route. To support the project, the CorrelAid team wrote a Python program that uses machine learning to classify road surface and pavement quality based on road images.

Use our API
------------
Send a GET request to [https://correlaid.cargorocket.de/predict?mapillary_keys=["{mapillary_key_one}", "{mapillary_key_two}]](https://correlaid.cargorocket.de/predict?mapillary_keys=["{mapillary_key_one}", "{mapillary_key_two}]), whereby you have to replace the placeholder with the Mapillary image IDs you would like to get smoothness and surface predictions for.

Details
------------
### Data processing
Data available via OpenStreetMap (OSM) regarding streets and bike lanes of Berlin were first linked to Mapillary photos to create our own dataset. Thereby, only streets / bike lanes with available smoothness and surface tags were taken as these information later served as information for training a neural network predicting these tags. We chose Berlin for developing our approach as the availability of tags, especially smoothness tags, was quite high there. 

You can find the whole code for creating a dataset in the `mapillary_image_classification/data/make_dataset.py` file. Unfortunately, this code was created for the version 3 of the Mapillary API. During the final phase of our project (07/2021), this API was not available anymore. After some time, it seemed to be online again; however, unfortunately, we cannot promise that this code is still usable. Therefore, you may have to adapt it to the Mapillary v4 API, if you want to generate a dataset using other OSM data.

For a better routing for cargo bikes, it is sufficient to know whether a certain street / bike lane has a good, intermediate or bad quality. That is why, we mapped the original surface and smoothness categories of OSM to our own categories. This enables an easier training of the Deep Learning model. Our final categories are the following
* **Surface:** paved, unpaved, cobblestone
* **Smoothness:** good, okay, bad
You can find the category mapping in the file `osm.py`.


### Deep Learning model generation
Based on this dataset, machine learning methods were used to create a classification model that can automatically determine the appropriate classification of surface type and quality for a given photo of a road. The model - a neural network - is available for prediction using the API request, mentioned above. 

For the preparation of the image data (done in each training epoch): The lower third was cut out of each image, because the upper part of the image often shows road surroundings like adjacent houses. The images were scaled to equal number of pixels, and randomly mirrored to get more variance. The preprocessing steps are implemented in the file `mapillary_image_classification/models/preprocessing.py`. The file `mapillary_image_classification/models/train_model.py` implements the actual training process using the Pytorch lightning library.

The model used was a MobileNet V3 adapted to multi-task learning with 2 labels (surface type and quality) and trained using 70 epochs. The best model reached a performance of 92% accuracy for the surface type prediction and an accuracy of 84% for predicting the smoothness. 

### API
The implementation details for the prediction API using Flask are available in the file `mapillary_image_classification/models/predict_model.py`

Setting up
------------
* Setup virtual environment and install dependencies listed in `requirements.txt`
* Copy `.env.example` to `.env` and set the variables

Further ideas
-------------
* Using the Mapillary object detections for cropping the images (as the object detections were not available over the Mapillary v4 API at first, we did not follow this idea anymore and used a static cropping process)
* Differentiating street and bike lanes in an image and predict their surface / smoothness separately

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
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
    ├── mapillary_image_classification                <- Source code for use in this project.
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
