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
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, f1_score, classification_report
import pprint
import json


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='data/processed/')
@click.argument('output_filepath', type=click.Path(), default='models/')
def main(input_filepath, output_filepath):
    """ Runs modelling scripts to turn preprocessed data from (../processed) into
        to a model (../models)
    """
    logger = logging.getLogger(__name__)
    
    INPUT_DIR = Path.cwd().joinpath(input_filepath)
    logger.info(f'modelling with data in {INPUT_DIR.name}')

    logger.info('loading envs')

    # cols
    BINARY_COLS = Path.cwd().joinpath(input_filepath).joinpath('binary-cols.pickle')
    CATEGORICAL_COLS = Path.cwd().joinpath(input_filepath).joinpath('categorical-cols.pickle')
    CONTINUOUS_COLS = Path.cwd().joinpath(input_filepath).joinpath('continuous-cols.pickle')
    TARGET_COL = Path.cwd().joinpath(input_filepath).joinpath('target-col.pickle')

    BINARY_COLS_OUT = Path.cwd().joinpath(output_filepath).joinpath('binary-cols.pickle')
    CATEGORICAL_COLS_OUT = Path.cwd().joinpath(output_filepath).joinpath('categorical-cols.pickle')
    CONTINUOUS_COLS_OUT = Path.cwd().joinpath(output_filepath).joinpath('continuous-cols.pickle')
    TARGET_COL_OUT = Path.cwd().joinpath(output_filepath).joinpath('target-col.pickle')

    COL_ORDER_OUT = Path.cwd().joinpath(output_filepath).joinpath('col-order.pickle')

    # data
    TRAIN_CSV = Path.cwd().joinpath(input_filepath).joinpath('train.csv')
    VAL_CSV = Path.cwd().joinpath(input_filepath).joinpath('val.csv')
    TEST_CSV = Path.cwd().joinpath(input_filepath).joinpath('test.csv')

    # metadata
    BINARY_ENCODERS = Path.cwd().joinpath(input_filepath).joinpath('binary-encoders.pickle')
    CATEGORICAL_ENCODERS = Path.cwd().joinpath(input_filepath).joinpath('categorical-encoders.pickle')
    TARGET_ENCODERS = Path.cwd().joinpath(input_filepath).joinpath('target-encoders.pickle')
    CONTINUOUS_SCALERS = Path.cwd().joinpath(input_filepath).joinpath('continuous-scalers.pickle')

    BINARY_ENCODERS_OUT = Path.cwd().joinpath(output_filepath).joinpath('binary-encoders.pickle')
    CATEGORICAL_ENCODERS_OUT = Path.cwd().joinpath(output_filepath).joinpath('categorical-encoders.pickle')
    TARGET_ENCODERS_OUT = Path.cwd().joinpath(output_filepath).joinpath('target-encoders.pickle')
    CONTINUOUS_SCALERS_OUT = Path.cwd().joinpath(output_filepath).joinpath('continuous-scalers.pickle')

    # model
    MODEL = Path.cwd().joinpath(output_filepath).joinpath('catboost_model.dump')

    # model results
    VAL_RESULTS = Path.cwd().joinpath(output_filepath).joinpath('val-results.txt')
    TEST_RESULTS = Path.cwd().joinpath(output_filepath).joinpath('test-results.txt')
    
    logger.info('loading data')
    
    # Cols
    binary_cols = read_obj(BINARY_COLS)
    categorical_cols = read_obj(CATEGORICAL_COLS)
    continuous_cols = read_obj(CONTINUOUS_COLS)
    target_col = read_obj(TARGET_COL)

    # Metadata
    ohe_encoders = read_obj(BINARY_ENCODERS)
    label_encoders = read_obj(CATEGORICAL_ENCODERS)
    scalers = read_obj(TARGET_ENCODERS)
    target_encoders = read_obj(CONTINUOUS_SCALERS)

    # Data
    X_train = pd.read_csv(TRAIN_CSV)
    X_val = pd.read_csv(VAL_CSV)
    X_test = pd.read_csv(TEST_CSV)
    
    X_train = X_train[binary_cols + categorical_cols + continuous_cols + [target_col]]
    X_val = X_val[binary_cols + categorical_cols + continuous_cols + [target_col]]
    X_test = X_test[binary_cols + categorical_cols + continuous_cols + [target_col]]
    
    y_train = X_train.pop(target_col)
    y_val = X_val.pop(target_col)
    y_test = X_test.pop(target_col)
    
    logger.info('modelling')
    model_args = {
        'custom_loss': ['AUC'],
        'random_seed': 42,
        'logging_level': 'Silent'
    }
    model = CatBoostClassifier(**model_args)

    cat_cols = binary_cols + categorical_cols
    model.fit(
        X_train, y_train,
        cat_features=cat_cols,
        logging_level='Verbose',
        plot=True
    )
    
    y_val_preds = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    y_val_proba_death = y_val_proba[:,1]
    val_results = results(y_val, y_val_preds, y_val_proba_death)
    #pprint.pprint(val_results)
    logger.info(f'val results: {val_results}')
    
    y_test_preds = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    y_test_proba_death = y_test_proba[:,1]
    test_results = results(y_test, y_test_preds, y_test_proba_death)
    logger.info(f'test results: {test_results}')
    
    logger.info('dumping data and saving model')
    dump_results(VAL_RESULTS, val_results)
    dump_results(TEST_RESULTS, test_results)
    
    # cols
    pickle_obj(BINARY_COLS_OUT, binary_cols)
    pickle_obj(CATEGORICAL_COLS_OUT, categorical_cols)
    pickle_obj(CONTINUOUS_COLS_OUT, continuous_cols)
    pickle_obj(TARGET_COL_OUT, target_col)
    pickle_obj(COL_ORDER_OUT, list(X_train.columns))

    # metadata
    pickle_obj(BINARY_ENCODERS_OUT, ohe_encoders)
    pickle_obj(CATEGORICAL_ENCODERS_OUT, label_encoders)
    pickle_obj(TARGET_ENCODERS_OUT, target_encoders)
    pickle_obj(CONTINUOUS_SCALERS_OUT, scalers)
    
    model.save_model(str(MODEL))
    
def read_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    return None

def results(y_true, y_pred, y_proba):
    results = {}
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['f1'] = f1_score(y_true, y_pred)
    fpr, tpr, thresh = roc_curve(y_true, y_proba)
    results['auc'] = auc(fpr, tpr)
    results['classification_report'] = classification_report(y_true, y_pred, output_dict=True)
    return results

def dump_results(path, results_dict):
    with open(path, 'w') as f:
        f.write(json.dumps(results_dict))
        
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
