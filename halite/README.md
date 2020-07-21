halite
==============================

Created by Two Sigma in 2016, more than 15,000 people around the world have participated in a Halite challenge. Players apply advanced algorithms in a dynamic, open source game setting. The strategic depth and immersive, interactive nature of Halite games make each challenge a unique learning environment.

The challenge is to create an agent that can succeed in the game of Halite IV.  (Kaggle Proj) https://www.kaggle.com/c/halite

# How-to Pull Episode Replays
This can module can be run to scrape episode replays from Kaggle.
These episode replays will be exported to "data/raw" as JSON.
This scraper respects the 60 request/minute rate-limit.
It can be configured further by "# How-to Configure" section.
Source code is available in `./halite_agent/data/fetch_dataset.py`.
Notebook used to develop the code is in `./notebooks/1-2-iwong-scraper.ipynb`.

```bash
$ conda create -n testenv python=3.6
$ conda activate testenv
$ pip install halite-agent
$ python3 -m halite_agent.data.fetch_dataset data/raw
```

# How-to Configure

```
HALITE_AGENT_FETCH_DATA_EPISODE_WATERMARK=1100 # pull games with avg. score > 1100
HALITE_AGENT_FETCH_DATA_TEAM_WATERMARK=25 # pull teams with rank > 25
HALITE_AGENT_FETCH_DATA_REQUEST_LIMIT=10 # run scraper that can make only 10 requests
HALITE_AGENT_FETCH_DATA_DISCOVERY_BUDGET=0.1 # used 10% of requests to query for new team metadata
HALITE_AGENT_FETCH_DATA_SCRAPER_METADATA_FILEPATH='./scraper_metadata' # store scraper metadata files here
HALITE_AGENT_FETCH_DATA_ARBITRARY_TEAM_ID='5118174' # use this team id to bootstrap the scraper
```

# How-to Develop

```bash
$ git clone https://github.com/iainwo/kaggle.git
$ cd halite/
$ make create_environment
$ conda activate halite
$ make requirements
$ vim ...
```

# Other Commands
```sh
(my-kaggle-project) talisman-2:my-kaggle-project iainwong$ make
Available rules:

build               Build python package 
clean               Delete all compiled Python files 
create_environment  Set up python interpreter environment 
data                Make Dataset 
data_final          Make Dataset for Kaggle Submission 
eda                 Generate visuals for feature EDA 
lint                Lint using flake8 
model               Make Model 
predictions         Make Predictions 
publish             Publish python package to PyPi 
requirements        Install Python Dependencies 
requirements_dev    Install Development Deps 
sync_data_from_s3   Download Data from S3 
sync_data_to_s3     Upload Data to S3 
test                Run unit tests 
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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
