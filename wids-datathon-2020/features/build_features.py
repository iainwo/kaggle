# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from pathlib import Path
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import bisect
import numpy as np
from itertools import combinations


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    
    INPUT_DIR = Path.cwd().joinpath(input_filepath)
    logger.info(f'feature-engineering with data in {INPUT_DIR.name}')

    logger.info('reading envs')
    # cols
    BINARY_COLS = Path.cwd().joinpath(input_filepath).joinpath('binary-cols.pickle')
    CATEGORICAL_COLS = Path.cwd().joinpath(input_filepath).joinpath('categorical-cols.pickle')
    CONTINUOUS_COLS = Path.cwd().joinpath(input_filepath).joinpath('continuous-cols.pickle')
    TARGET_COL = Path.cwd().joinpath(input_filepath).joinpath('target-col.pickle')

    BINARY_COLS_OUT = Path.cwd().joinpath(output_filepath).joinpath('binary-cols.pickle')
    CATEGORICAL_COLS_OUT = Path.cwd().joinpath(output_filepath).joinpath('categorical-cols.pickle')
    CONTINUOUS_COLS_OUT = Path.cwd().joinpath(output_filepath).joinpath('continuous-cols.pickle')
    TARGET_COL_OUT = Path.cwd().joinpath(output_filepath).joinpath('target-col.pickle')

    # data
    TRAIN_CSV = Path.cwd().joinpath(input_filepath).joinpath('train.csv')
    VAL_CSV = Path.cwd().joinpath(input_filepath).joinpath('val.csv')
    TEST_CSV = Path.cwd().joinpath(input_filepath).joinpath('test.csv')

    TRAIN_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('train.csv')
    VAL_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('val.csv')
    TEST_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('test.csv')

    # metadata
    BINARY_ENCODERS = Path.cwd().joinpath(output_filepath).joinpath('binary-encoders.pickle')
    CATEGORICAL_ENCODERS = Path.cwd().joinpath(output_filepath).joinpath('categorical-encoders.pickle')
    TARGET_ENCODERS = Path.cwd().joinpath(output_filepath).joinpath('target-encoders.pickle')
    CONTINUOUS_SCALERS = Path.cwd().joinpath(output_filepath).joinpath('continuous-scalers.pickle')
    CONTINUOUS_FILLERS = Path.cwd().joinpath(output_filepath).joinpath('continuous-fillers.pickle')
    
    logger.info('reading in data')
    binary_cols = read_obj(BINARY_COLS)
    categorical_cols = read_obj(CATEGORICAL_COLS)
    continuous_cols = read_obj(CONTINUOUS_COLS)
    target_col = read_obj(TARGET_COL)

    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)
    
    logger.info('synthesizing categorical pairs')
    pair_cols = [
        'ethnicity',
        'gender',
        'hospital_admit_source',
        'icu_admit_source',
        'icu_stay_type',
        'icu_type',
        'apache_3j_bodysystem',
        'apache_2_bodysystem'
    ]
    cmbs = list(combinations(pair_cols, 2))
    combo_cols = list()
    for cols in cmbs:
        col_name = f'paired_{"_".join(cols)}'
        combo_cols.append(col_name)
        train[col_name] = concat_columns(train, cols)
        val[col_name] = concat_columns(val, cols)
        test[col_name] = concat_columns(test, cols)
    categorical_cols.extend(combo_cols)
    
    logger.info('synthesizing categorical triplets')
    cmbs = list(combinations(pair_cols, 3))
    combo_cols = list()
    for cols in cmbs:
        col_name = f'paired_{"_".join(cols)}'
        combo_cols.append(col_name)
        train[col_name] = concat_columns(train, cols)
        val[col_name] = concat_columns(val, cols)
        test[col_name] = concat_columns(test, cols)
    categorical_cols.extend(combo_cols)

    logger.info('synthesizing binary pairs')
    combo_cols = list()
    for cols in list(combinations(binary_cols, 2)):
        col_name = f'paired_{"_".join(cols)}'
        combo_cols.append(col_name)
        train[col_name] = concat_columns(train, cols)
        val[col_name] = concat_columns(val, cols)
        test[col_name] = concat_columns(test, cols)
    categorical_cols.extend(combo_cols)
    
    # aggregate icu
    train['hospital_admit_source_is_icu'] = train['hospital_admit_source'].apply(
            lambda x: 
                'True' if x in [
                    'Other ICU', 
                    'ICU to SDU',
                    'ICU'
                ] else 'False')
    val['hospital_admit_source_is_icu'] = val['hospital_admit_source'].apply(
            lambda x: 
                'True' if x in [
                    'Other ICU', 
                    'ICU to SDU',
                    'ICU'
                ] else 'False')
    test['hospital_admit_source_is_icu'] = test['hospital_admit_source'].apply(
            lambda x: 
                'True' if x in [
                    'Other ICU', 
                    'ICU to SDU',
                    'ICU'
                ] else 'False')
    categorical_cols.append('hospital_admit_source_is_icu')
    
    # aggregate ethnicity
    common_cols = [np.nan, 'Other/Unknown']
    train['ethnicity_is_unknown'] = train['ethnicity'].apply(lambda x: True if x in common_cols else False)
    val['ethnicity_is_unknown'] = val['ethnicity'].apply(lambda x: True if x in common_cols else False)
    test['ethnicity_is_unknown'] = test['ethnicity'].apply(lambda x: True if x in common_cols else False)
    categorical_cols.append('ethnicity_is_unknown')
    
    # aggregate cardiac
    common_cols = ['CTICU', 'CCU-CTICU', 'Cardiac ICU', 'CSICU']
    train['icu_type_is_cardiac'] = train['icu_type'].apply(lambda x: True if x in common_cols else False)
    val['icu_type_is_cardiac'] = val['icu_type'].apply(lambda x: True if x in common_cols else False)
    test['icu_type_is_cardiac'] = test['icu_type'].apply(lambda x: True if x in common_cols else False)
    categorical_cols.append('icu_type_is_cardiac')
    
    # aggregate apache_2_bodysystem
    common_cols = ['Undefined Diagnoses', np.nan, 'Undefined diagnoses']
    train['apache_2_bodysystem_is_undefined'] = train['apache_2_bodysystem'].apply(lambda x: True if x in common_cols else False)
    val['apache_2_bodysystem_is_undefined'] = val['apache_2_bodysystem'].apply(lambda x: True if x in common_cols else False)
    test['apache_2_bodysystem_is_undefined'] = test['apache_2_bodysystem'].apply(lambda x: True if x in common_cols else False)
    categorical_cols.append('apache_2_bodysystem_is_undefined')
    
    logger.info('typifying...')
    # continuous
    train[continuous_cols] = train[continuous_cols].astype('float32')
    val[continuous_cols] = val[continuous_cols].astype('float32')
    test[continuous_cols] = test[continuous_cols].astype('float32')
    
    # cats
    train[categorical_cols] = train[categorical_cols].astype('str').astype('category')
    val[categorical_cols] = val[categorical_cols].astype('str').astype('category')
    test[categorical_cols] = test[categorical_cols].astype('str').astype('category')
    
    # binary
    train[binary_cols] = train[binary_cols].astype('str').astype('category')
    val[binary_cols] = val[binary_cols].astype('str').astype('category')
    test[binary_cols] = test[binary_cols].astype('str').astype('category')
    
    # targets
    train[target_col] = train[target_col].astype('str').astype('category')
    val[target_col] = val[target_col].astype('str').astype('category')
    test[target_col] = test[target_col].astype('str').astype('category')
    
    logger.info("dropping")
    train = train.dropna(how='all')
    val = val.dropna(how='all')
    test = test.dropna(how='all')
    
    logger.info('dropping confounding features')
    confounding_cols = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']

    # remove confounding vars - biases and undue variance
    for x in confounding_cols:
        continuous_cols.remove(x)
    train = train.drop(columns=confounding_cols)
    val = val.drop(columns=confounding_cols)
    test = test.drop(columns=confounding_cols)
    
    train = train[[target_col] + continuous_cols + categorical_cols + binary_cols]
    val = val[[target_col] + continuous_cols + categorical_cols + binary_cols]
    test = test[[target_col] + continuous_cols + categorical_cols + binary_cols]
      
    logger.info('filling')
    train, fillers = fill(train, continuous_cols)
    val, _ = fill(val, continuous_cols, fillers)
    test, _ = fill(test, continuous_cols, fillers)
    categorical_cols.extend([f'{x}_na' for x in continuous_cols])
    
    logger.info('normalizing')
    train, scalers = normalize(train, continuous_cols)
    val, _ = normalize(val, continuous_cols, scalers)
    test, _ = normalize(test, continuous_cols, scalers)
    
    logger.info('label encoding categoricals')
    
    # to handle converting fill_nas from bool to str
    train[categorical_cols] = train[categorical_cols].astype('str').astype('category')
    val[categorical_cols] = val[categorical_cols].astype('str').astype('category')
    test[categorical_cols] = test[categorical_cols].astype('str').astype('category')
    
    train, label_encoders = labelencode(train, categorical_cols)
    val, _ = labelencode(val, categorical_cols, label_encoders)
    test, _ = labelencode(test, categorical_cols, label_encoders)
    
    logger.info('label encoding targets')
    train, target_encoders = labelencode(train, [target_col])
    val, _ = labelencode(val, [target_col], target_encoders)
    test, _ = labelencode(test, [target_col], target_encoders)
    
    logger.info('one hot encoding categoricals')
    # TODO: change to ohe later
    train, ohe_encoders = labelencode(train, binary_cols)
    val, _ = labelencode(val, binary_cols, ohe_encoders)
    test, _ = labelencode(test, binary_cols, ohe_encoders)

    logger.info('dropping identifiers from list cols')
    for col in ['encounter_id', 'hospital_id', 'patient_id', 'icu_id']:
        continuous_cols.remove(col)

    logger.info('persisting data')    
    # cols
    pickle_obj(BINARY_COLS_OUT, binary_cols)
    pickle_obj(CATEGORICAL_COLS_OUT, categorical_cols)
    pickle_obj(CONTINUOUS_COLS_OUT, continuous_cols)
    pickle_obj(TARGET_COL_OUT, target_col)

    # metadata
    pickle_obj(BINARY_ENCODERS, ohe_encoders)
    pickle_obj(CATEGORICAL_ENCODERS, label_encoders)
    pickle_obj(TARGET_ENCODERS, target_encoders)
    pickle_obj(CONTINUOUS_SCALERS, scalers)
    pickle_obj(CONTINUOUS_FILLERS, fillers)

    # data
    train.to_csv(TRAIN_CSV_OUT, index=False)
    val.to_csv(VAL_CSV_OUT, index=False)
    test.to_csv(TEST_CSV_OUT, index=False)
    
def read_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    return None

def concat_columns(df, columns):
    value = df[columns[0]].astype(str) + ' '
    for col in columns[1:]:
        value += df[col].astype(str) + ' '
    return value

def fill(df, cols, fillers=None):
    if None is fillers:
        fillers = dict()
        
        for col in cols:
            logging.info(f'generating fill col {col}')
            for hospital_id in df['hospital_id'].unique():

                subset = df[df['hospital_id'] == hospital_id]
                hospital_col_key = f'{col}_{hospital_id}'

                if hospital_col_key not in fillers:
                    fillers[hospital_col_key] = subset[col].dropna().median()
            
            fillers[col] = df[col].dropna().median()
    
    for col in cols:
        logging.info(f'fillling {col}')
        for hospital_id in df['hospital_id'].unique():
            subset = df[df['hospital_id'] == hospital_id]
            hospital_col_key = f'{col}_{hospital_id}'
            
            if hospital_col_key in fillers:
                df.loc[df['hospital_id'] == hospital_id, col] = subset[col].fillna(fillers[hospital_col_key])
            else:
                df.loc[df['hospital_id'] == hospital_id, col] = subset[col].fillna(fillers[col])
            
            df.loc[df['hospital_id'] == hospital_id, f'{col}_na'] = pd.isnull(subset[col])
    
    return df, fillers
# def fill(df, cols, fillers=None):
#     if None is fillers:
#         fillers = dict()
#     for col in cols:
#         if col not in fillers:
#             fillers[col] = df[col].dropna().median()
            
#         df[f'{col}_na'] = pd.isnull(df[col])
#         df[col] = df[col].fillna(fillers[col])
    
#     return df, fillers

def normalize(df, cols, scalers=None):
    if None is scalers:
        scalers = dict()
        
    for col in cols:
        if col not in scalers:
            scalers[col] = StandardScaler(with_mean=True, with_std=True)
            scalers[col].fit(df[col].values.reshape(-1,1))
        
        scaler = scalers[col]
        df[col] = scaler.transform(df[col].values.reshape(-1,1))
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

def pickle_obj(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
