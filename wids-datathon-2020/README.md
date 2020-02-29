wids_datathon_2020
==============================

The challenge is to create a model that uses data from the first 24 hours of intensive care to predict patient survival. (Kaggle Proj) https://www.kaggle.com/c/widsdatathon2020/overview

# How to use

```bash
$ make create_environment
$ conda activate wids_datathon_2020
$ make requirements # will also install pip env to ipykernel for jupyterlab/hub
$ make requirements_dev # to install dev reqs

# 1. preprocess data - unzips the data/external into data/raw, then creates train-test-split into data/interim
# 2. feature engineer - normalizes, label encodes, fills, feature engineers
$ make data

$ make eda # generate eda analysis (reports/eda/)

$ make model # output to ./models/

$ make predictions # output to ./data/predictions/
```

# Other Commands
```sh
(wids_datathon_2020) iainwong@talisman-2:Development/wids_datathon_2020 [master●] $ make                                                                                
Available rules:

clean               Delete all compiled Python files 
create_environment  Set up python interpreter environment 
data                Make Dataset 
lint                Lint using flake8 
model               Make Model 
predictions         Make Predictions 
requirements        Install Python Dependencies 
requirements_dev    Install Development Deps 
sync_data_from_s3   Download Data from S3 
sync_data_to_s3     Upload Data to S3 
test_environment    Test python environment is setup correctly 
```

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
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="../kaggle-data-science/">kaggle-data-science</a> project template.</small></p>
