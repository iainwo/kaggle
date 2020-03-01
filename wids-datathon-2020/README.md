wids_datathon_2020
==============================

The challenge is to create a model that uses data from the first 24 hours of intensive care to predict patient survival. (Kaggle Proj) https://www.kaggle.com/c/widsdatathon2020/overview

# How-to Use Perform Inference
This project provides a publicly-accessible and straight forward way to perform batch or realtime inference based on WiDS Datathon 2020 data.

There are essentially four steps required for inference:

1. Obtain a copy of the [Kaggle Competition Dataset](https://www.kaggle.com/c/widsdatathon2020/data)
2. Obtain a copy or fabricate data which to perform inference upon.
3. Use the modelling `wids-datathon-2020` PyPi module to create a model and inference-requisite preprocessing artifacts
4. Apply the preprocessing artifacts and model to the inference data to manufacture batch or realtime inference

## 1. Obtain a copy of the [Kaggle Competition Dataset](https://www.kaggle.com/c/widsdatathon2020/data)

```bash
$ mkdir -p data/external data/raw data/interim data/processed data/predictions models/
$ wget -O data/external/widsdatathon2020.zip https://github.com/iainwo/kaggle/blob/master/wids-datathon-2020/data/external/widsdatathon2020.zip
```

## 2. Obtain a copy or fabricate data which to perform inference upon.

```bash
$ touch data/raw/my-inference-samples.csv
```

## 3. Use the modelling `wids-datathon-2020` PyPi module to create a model and inference-requisite preprocessing artifacts

```bash
$ echo "Prepare software env"
$ conda create -n testenv python=3.6
$ conda activate testenv
$ pip install wids-datathon-2020

$ echo "Stage data"
$ mkdir -p data/external data/raw data/interim data/processed data/predictions models/
$ zip widsdatathon2020.zip "WiDS Datathon 2020 Dictionary.csv" training_v2.csv unlabeled.csv
$ cp widsdatathon2020.zip data/external

$ echo "Model predictions"
$ python3 -m wids-datathon-2020.data.make_dataset data/raw data/interim
$ python3 -m wids-datathon-2020.features.build_features data/interim data/processed
$ python3 -m wids-datathon-2020.models.train_model data/processed models/
$ python3 -m wids-datathon-2020.models.predict_model models/ data/processed/ data/predictions

$ echo "Observe model and preprocessing artifacts"
$ ls -larth models/
$ ls -larth data/predictions/
```

## 4. Apply the preprocessing artifacts and model to the inference data to manufacture batch or realtime inference

Refer to [this notebook](./notebooks/5.0.0-iwong-batch-prediction.ipynb) for a cell-by-cell example.
At a high-level realtime inference would look something like this:

```python
df = pd.read_csv('my-inference-samples.csv')

# cast
df[continuous_cols] = df[continuous_cols].astype('float32')
df[categorical_cols] = df[categorical_cols].astype('str').astype('category')
df[binary_cols] = df[binary_cols].astype('str').astype('category')
df[target_col] = df[target_col].astype('str').astype('category')

# fill
df[continuous_cols] = df[continuous_cols].fillna(0)

# normalize, labelencode, ohe
df, _ = normalize(df, continuous_cols, scalers)
# ...

y_preds = model.predict(X)
y_proba = model.predict_proba(X)
y_proba_death = y_proba[:,1]

```

# How-to Develop

```bash
$ echo 'setup development environment'
$ git clone https://github.com/iainwo/kaggle.git
$ cd wids-datathon-2020/
$ make create_environment
$ conda activate wids_datathon_2020
$ make requirements

$ echo 'make some changes to the wids-datathon-2020 python module'
$ vim my-file.py

$ echo 'run module'
$ make data
$ make model
$ make predictions
```

# Other Commands
```sh
(wids_datathon_2020) talisman-2:wids-datathon-2020 iainwong$ make
Available rules:

clean               Delete all compiled Python files 
create_environment  Set up python interpreter environment 
data                Make Dataset 
data_final          Make Dataset for Kaggle Submission 
eda                 Generate visuals for feature EDA 
lint                Lint using flake8 
model               Make Model 
predictions         Make Predictions 
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

<p><small>Project based on the <a target="_blank" href="../kaggle-data-science/">kaggle-data-science</a> project template.</small></p>
