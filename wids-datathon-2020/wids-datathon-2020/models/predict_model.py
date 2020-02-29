# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from catboost import CatBoostClassifier


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='models/')
@click.argument('inference_filepath', type=click.Path(exists=True), default='data/processed/')
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
    PRED_CSV = Path.cwd().joinpath(inference_filepath).joinpath('preds.csv')
    PRED_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('preds.csv')

    # model
    MODEL = Path.cwd().joinpath(input_filepath).joinpath('catboost_model.dump')

    logger.info('loading data')
    df = pd.read_csv(PRED_CSV)

    # Store the encounter ids, before removing them for training
    arr = df['encounter_id']

    # Model
    model = CatBoostClassifier()
    model.load_model(str(MODEL))

    X = df[model.feature_names_]

    logger.info('making predictions')
    y_proba = model.predict_proba(X)
    y_proba_death = y_proba[:, 1]

    y = pd.DataFrame(y_proba_death, columns=['hospital_death']).astype('float32')

    logger.info('persisting predictions')
    # arr = scalers['encounter_id'].inverse_transform(X['encounter_id'])
    X_encounter_id = round(pd.DataFrame(arr, columns=['encounter_id']))  # round for numerical errs
    X_encounter_id = X_encounter_id.astype('int32')

    pd.concat([X_encounter_id, y], axis=1).to_csv(PRED_CSV_OUT, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
