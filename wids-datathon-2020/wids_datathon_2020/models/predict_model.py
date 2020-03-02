# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from catboost import CatBoostClassifier


def predict_model(inference_filepath, model_filepath, output_filepath):
    """ Runs modelling scripts to turn preprocessed data from (../processed) into
        to a model (../models)
    """
    logger = logging.getLogger(__name__)

    logger.info(f'loading model {model_filepath}')
    model = CatBoostClassifier().load_model(str(Path.cwd().joinpath(model_filepath)))

    logger.info(f'loading inference data {inference_filepath}')
    df = pd.read_feather(Path.cwd().joinpath(inference_filepath))

    # Store the encounter ids, before removing them for training
    arr = df['encounter_id']

    logger.info(f'inference data has cols: {df.columns}')
    logger.info(f'masking inference data to cols: {model.feature_names_}')
    X = df[model.feature_names_]

    logger.info('making predictions')
    y_proba = model.predict_proba(X)
    y_proba_death = y_proba[:, 1]

    y = pd.DataFrame(y_proba_death, columns=['hospital_death']).astype('float32')

    logger.info('persisting predictions')
    # arr = scalers['encounter_id'].inverse_transform(X['encounter_id'])
    X_encounter_id = round(pd.DataFrame(arr, columns=['encounter_id']))  # round for numerical errs
    X_encounter_id = X_encounter_id.astype('int32')

    output_filename = Path.cwd().joinpath(inference_filepath).stem + '.csv'
    pd.concat([X_encounter_id, y], axis=1).to_csv(Path.cwd().joinpath(output_filepath).joinpath(output_filename), index=False)


@click.command()
@click.argument('inference_filepath', type=click.Path(exists=True), default='data/processed/unlabeled_encoded.feather')
@click.argument('model_filepath', type=click.Path(exists=True), default='models/model.dump')
@click.argument('output_filepath', type=click.Path(), default='data/predictions/')
def main(inference_filepath, model_filepath, output_filepath):
    predict_model(inference_filepath, model_filepath, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
