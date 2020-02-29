# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np


@click.command()
@click.option('--is-final-model', is_flag=True)
@click.option('--is-semi-supervised-model', is_flag=True)
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('external_filepath', type=click.Path(exists=True), default='data/external/')
def main(is_final_model, is_semi_supervised_model, input_filepath, output_filepath, external_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    INPUT_FILEPATH = Path.cwd().joinpath(input_filepath)
    DATASET = Path.cwd().joinpath(external_filepath).joinpath('widsdatathon2020.zip')

    TRAIN_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('train.csv')
    VAL_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('val.csv')
    TEST_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('test.csv')
    PREDS_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('preds.csv')

    DATADICT_FILEPATH = Path.cwd().joinpath(
        input_filepath).joinpath('WiDS Datathon 2020 Dictionary.csv')
    DATA = Path.cwd().joinpath(input_filepath).joinpath('training_v2.csv')
    PREDICTION_DATA = Path.cwd().joinpath(input_filepath).joinpath('unlabeled.csv')
    SEMI_SUPERVISED_DATA = Path.cwd().joinpath(
        input_filepath).joinpath('semi-supervised.csv')
    # REGRESSIVE_PROB_DATA = Path.cwd().joinpath(input_filepath).joinpath('regressive_death_prob.csv')

    with zipfile.ZipFile(DATASET, 'r') as zip_ref:
        zip_ref.extractall(INPUT_FILEPATH)

    # Get Col Types
    logger.info('getting col types...')
    datadict = pd.read_csv(DATADICT_FILEPATH)

    continuous_cols = list(
        list(datadict[datadict['Data Type'] == 'integer']
             ['Variable Name'].unique())
        + list(datadict[datadict['Data Type'] == 'numeric']
               ['Variable Name'].unique())
    )
    categorical_cols = list(
        datadict[datadict['Data Type'] == 'string']['Variable Name'].unique())
    binary_cols = list(
        datadict[datadict['Data Type'] == 'binary']['Variable Name'].unique())

    target_col = 'hospital_death'
    binary_cols.remove(target_col)

    # fix datadict variable names
    categorical_cols.remove('icu_admit_type')
    continuous_cols.remove('pred')

    # fix missmapped continuous cols
    continuous_cols.extend(
        ['bmi', 'apache_2_diagnosis', 'apache_3j_diagnosis'])
    categorical_cols = [x for x in categorical_cols if x not in [
        'bmi', 'apache_2_diagnosis', 'apache_3j_diagnosis']]

    logger.info(f'Target cols: {target_col}')
    logger.info(f'Continuous cols: {continuous_cols}')
    logger.info(f'Categorical cols: {categorical_cols}')
    logger.info(f'Binary cols: {binary_cols}')

    # Get data
    logger.info('getting data...')
    df = pd.read_csv(DATA)
    preds = pd.read_csv(PREDICTION_DATA)
    if is_semi_supervised_model:
        semi = pd.read_csv(SEMI_SUPERVISED_DATA)
    # prob = pd.read_csv(REGRESSIVE_PROB_DATA)

    # df.loc[df['encounter_id'].isin(prob['encounter_id']),
    # 'apache_4a_hospital_death_prob'] =
    #  prob.loc[prob['encounter_id'].isin(df['encounter_id']),
    # 'apache_4a_hospital_death_prob'].values

    df = df[[target_col] + continuous_cols + categorical_cols + binary_cols]
    preds = preds[continuous_cols + categorical_cols + binary_cols]

    logger.info(f'Dataframe has these cols {list(df.columns)}')

    # Typify
    logger.info('typifying...')
    df[continuous_cols] = df[continuous_cols].astype('float32')
    df[categorical_cols] = df[categorical_cols].astype(
        'str').astype('category')
    df[binary_cols] = df[binary_cols].astype('str').astype('category')
    df[target_col] = df[target_col].astype('str').astype('category')

    preds[continuous_cols] = preds[continuous_cols].astype('float32')
    preds[categorical_cols] = preds[categorical_cols].astype(
        'str').astype('category')
    preds[binary_cols] = preds[binary_cols].astype('str').astype('category')

    # Splitting
    strat_cols = [target_col]
    logger.info(
        f'doing test-train-split stratifying on cols: {strat_cols} and hospital_id')

    strat_cols = [target_col]

    train = pd.DataFrame()
    val = pd.DataFrame()
    test = pd.DataFrame()

    for hospital_id in df['hospital_id'].unique():
        subset = df[df['hospital_id'] == hospital_id]

        if 10 < len(subset):
            classes, y_indices = np.unique(
                subset[strat_cols],
                return_inverse=True
            )
            class_counts = np.bincount(y_indices)

            if 2 > np.min(class_counts):
                tmp_train, tmp_test = train_test_split(
                    subset,
                    test_size=0.1,
                    random_state=0
                )
                tmp_train, tmp_val = train_test_split(
                    tmp_train,
                    test_size=0.1,
                    random_state=0
                )
            else:
                tmp_train, tmp_test = train_test_split(
                    subset,
                    test_size=0.1,
                    random_state=0,
                    stratify=subset[strat_cols]
                )
                tmp_train, tmp_val = train_test_split(
                    tmp_train,
                    test_size=0.1,
                    random_state=0,
                    stratify=tmp_train[strat_cols]
                )

            train = train.append(tmp_train)
            val = val.append(tmp_val)
            test = test.append(tmp_test)

    if is_semi_supervised_model:
        logger.info('Using semi-supervised data')
        semi_preds = preds[preds['encounter_id'].isin(semi['encounter_id'])]
        semi_preds[target_col] = 1
        df = df.append(semi_preds)
        train = train.append(semi_preds)

    # Persist data
    if is_final_model:
        logger.info('Using all data')
        df.to_csv(TRAIN_CSV_OUT, index=False)
    else:
        logger.info('Using train data')
        train.to_csv(TRAIN_CSV_OUT, index=False)
    val.to_csv(VAL_CSV_OUT, index=False)
    test.to_csv(TEST_CSV_OUT, index=False)
    preds.to_csv(PREDS_CSV_OUT, index=False)

    with open(Path.cwd()
              .joinpath(output_filepath)
              .joinpath('continuous-cols.pickle'), 'wb') as f:
        pickle.dump(continuous_cols, f)

    with open(Path.cwd()
              .joinpath(output_filepath)
              .joinpath('target-col.pickle'), 'wb') as f:
        pickle.dump(target_col, f)

    with open(Path.cwd()
              .joinpath(output_filepath)
              .joinpath('categorical-cols.pickle'), 'wb') as f:
        pickle.dump(categorical_cols, f)

    with open(Path.cwd()
              .joinpath(output_filepath)
              .joinpath('binary-cols.pickle'), 'wb') as f:
        pickle.dump(binary_cols, f)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
