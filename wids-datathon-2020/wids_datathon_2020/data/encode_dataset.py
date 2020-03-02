# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import bisect
import numpy as np


def encode_dataset(input_filepath, output_filepath, encoder_filepath, is_create_encoders):
    """ Encode the dataset: encoding, scaling and filling the
        continuous and categorical values.
    """
    logger = logging.getLogger(__name__)

    FILLERS_PATH = Path.cwd().joinpath(encoder_filepath).joinpath('numerical-fillers.pickle')
    SCALERS_PATH = Path.cwd().joinpath(encoder_filepath).joinpath('numerical-scalers.pickle')
    ENCODERS_PATH = Path.cwd().joinpath(encoder_filepath).joinpath('categorical-encoders.pickle')

    logger.info(f'Encoding file {input_filepath}')
    df = pd.read_feather(Path.cwd().joinpath(input_filepath))

    metadata_cols = ['encounter_id']

    numerical_cols = df.select_dtypes(include='number').columns
    numerical_cols = [x for x in numerical_cols if x not in metadata_cols]
    logger.info(f'Enumerated numerical columns {numerical_cols}')

    categorical_cols = df.select_dtypes(include='category').columns
    categorical_cols = [x for x in categorical_cols if x not in metadata_cols]
    logger.info(f'Enumerated categorical columns {categorical_cols}')

    logger.info(f'Importing fillers, scalers and encoders')

    try:
        fillers = read_obj(FILLERS_PATH)
        scalers = read_obj(SCALERS_PATH)
        encoders = read_obj(ENCODERS_PATH)
    except FileNotFoundError:
        logger.warn('Fillers, scalers or encoders not found.')

    if is_create_encoders:
        logger.info(f'Creating new fillers, scalers, and encoders')
        tmp = df.copy()
        _, fillers = fill(tmp, numerical_cols)
        _, scalers = normalize(tmp, numerical_cols)
        _, encoders = labelencode(tmp, categorical_cols)

    logger.info(f'Filling numericals')
    df, _ = fill(df, numerical_cols, fillers)

    logger.info(f'Normalizing numericals')
    df, _ = normalize(df, numerical_cols, scalers)

    logger.info(f'Labelencoding categoricals')
    df, _ = labelencode(df, categorical_cols, encoders)

    if is_create_encoders:
        logger.info(f'Persisting fillers, scalers and encoders')
        pickle_obj(FILLERS_PATH, fillers)
        pickle_obj(SCALERS_PATH, scalers)
        pickle_obj(ENCODERS_PATH, encoders)

    logger.info(f'Persisting encoded dataset')
    output_filename = Path.cwd().joinpath(input_filepath).stem
    df.reset_index(drop=True).to_feather(Path.cwd().joinpath(output_filepath).joinpath(output_filename + '_encoded.feather'))


def pickle_obj(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def read_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    return None


def fill(df, cols, fillers=None, is_demarc=True, fill_val=None):
    if None is fillers:
        fillers = dict()
    for col in cols:
        if col not in fillers:
            if fill_val is not None:
                fillers[col] = fill_val
            else:
                fillers[col] = df[col].dropna().mean()

        if is_demarc:
            df[f'{col}_na'] = pd.isnull(df[col])
        df[col] = df[col].fillna(fillers[col])

    return df, fillers


def normalize(df, cols, scalers=None):
    if None is scalers:
        scalers = dict()

    for col in cols:
        if col not in scalers:
            scalers[col] = StandardScaler(with_mean=True, with_std=True)
            scalers[col].fit(df[col].values.reshape(-1, 1))

        scaler = scalers[col]
        df[col] = scaler.transform(df[col].values.reshape(-1, 1))
    return df, scalers


def labelencode(df, cols, encoders=None, unknown_value='UNK'):
    if None is encoders:
        encoders = dict()

    for col in cols:
        if col not in encoders:
            le = LabelEncoder()
            le.fit(df[col].values)

            # add unknown val to cats
            cats = le.classes_.tolist()
            bisect.insort_left(cats, unknown_value)

            # redefine cats on le
            le.classes_ = np.asarray(cats)

            encoders[col] = le

        le = encoders[col]
        df[col] = df[col].map(lambda x: unknown_value if x not in le.classes_ else x)
        df[col] = le.transform(df[col].values)

    return df, encoders


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='data/interim/training_v2_train.feather')
@click.argument('output_filepath', type=click.Path(exists=True), default='data/processed/')
@click.argument('encoder_filepath', type=click.Path(exists=True), default='models/')
@click.option('--is-create-encoders', is_flag=True)
def main(input_filepath, output_filepath, encoder_filepath, is_create_encoders):
    encode_dataset(input_filepath, output_filepath, encoder_filepath, is_create_encoders)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
