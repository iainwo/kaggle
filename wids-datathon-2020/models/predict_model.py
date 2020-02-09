# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
from pathlib import Path
import pickle
import numpy as np
from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import bisect

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='models/')
@click.argument('inference_filepath', type=click.Path(exists=True), default='data/raw/')
@click.argument('output_filepath', type=click.Path(), default='data/predictions/')
def main(input_filepath, inference_filepath, output_filepath):
    """ Runs modelling scripts to turn preprocessed data from (../processed) into
        to a model (../models)
    """
    logger = logging.getLogger(__name__)
    
    INPUT_DIR = Path.cwd().joinpath(input_filepath)
    logger.info(f'prediction with model in {INPUT_DIR.name}')
    
    logger.info('loading envs')

    # prediction data
    PRED_CSV = Path.cwd().joinpath(inference_filepath).joinpath('unlabeled.csv')
    PRED_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('preds.csv')

    # cols
    BINARY_COLS = Path.cwd().joinpath(input_filepath).joinpath('binary-cols.pickle')
    CATEGORICAL_COLS = Path.cwd().joinpath(input_filepath).joinpath('categorical-cols.pickle')
    CONTINUOUS_COLS = Path.cwd().joinpath(input_filepath).joinpath('continuous-cols.pickle')
    TARGET_COL = Path.cwd().joinpath(input_filepath).joinpath('target-col.pickle')

    COL_ORDER = Path.cwd().joinpath(input_filepath).joinpath('col-order.pickle')

    # metadata
    BINARY_ENCODERS = Path.cwd().joinpath(input_filepath).joinpath('binary-encoders.pickle')
    CATEGORICAL_ENCODERS = Path.cwd().joinpath(input_filepath).joinpath('categorical-encoders.pickle')
    TARGET_ENCODERS = Path.cwd().joinpath(input_filepath).joinpath('target-encoders.pickle')
    CONTINUOUS_SCALERS = Path.cwd().joinpath(input_filepath).joinpath('continuous-scalers.pickle')

    # model
    MODEL = Path.cwd().joinpath(input_filepath).joinpath('catboost_model.dump')
    
    logger.info('loading data')

    # Cols
    binary_cols = read_obj(BINARY_COLS)
    categorical_cols = read_obj(CATEGORICAL_COLS)
    continuous_cols = read_obj(CONTINUOUS_COLS)
    target_col = read_obj(TARGET_COL)

    col_order = read_obj(COL_ORDER)

    # Metadata
    ohe_encoders = read_obj(BINARY_ENCODERS)
    label_encoders = read_obj(CATEGORICAL_ENCODERS)
    scalers = read_obj(TARGET_ENCODERS)
    target_encoders = read_obj(CONTINUOUS_SCALERS)

    # Data
    df = pd.read_csv(PRED_CSV)
    df = df[binary_cols + categorical_cols + continuous_cols + [target_col]]

    # Model
    model = CatBoostClassifier()
    model.load_model(str(MODEL))

    logger.info('casting')    
    df[continuous_cols] = df[continuous_cols].astype('float32')
    df[categorical_cols] = df[categorical_cols].astype('str').astype('category')
    df[binary_cols] = df[binary_cols].astype('str').astype('category')
    df[target_col] = df[target_col].astype('str').astype('category')

    logger.info('filling')
    df[continuous_cols] = df[continuous_cols].fillna(0)

    logger.info('normalizing')
    df, _ = normalize(df, continuous_cols, scalers)
    
    logger.info('encoding')
    df, _ = labelencode(df, categorical_cols, label_encoders)
    df, _ = labelencode(df, [target_col], target_encoders)
    df, _ = labelencode(df, binary_cols, ohe_encoders)
    
    logger.info('setting column order')
    if target_col in col_order:
        col_order.remove(target_col)
    X = df[col_order]
    
    logger.info('making predictions')
    y_preds = model.predict(X)
    y_proba = model.predict_proba(X)
    y_proba_death = y_proba[:,1]
    
    y = pd.DataFrame(y_proba_death, columns=['hospital_death']).astype('float32')
    
    logger.info('persisting predictions')
    arr = scalers['encounter_id'].inverse_transform(X['encounter_id'])
    X_encounter_id = round(pd.DataFrame(arr, columns=['encounter_id'])) # round for numerical errs
    X_encounter_id = X_encounter_id.astype('int32')

    pd.concat([X_encounter_id, y], axis=1).to_csv(PRED_CSV_OUT, index=False)
    

def read_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    return None

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

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
