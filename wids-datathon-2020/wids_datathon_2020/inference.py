# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from pathlib import PurePath
from dotenv import find_dotenv, load_dotenv
from wids_datathon_2020.data.make_dataset import make_dataset
from wids_datathon_2020.data.encode_dataset import encode_dataset
from wids_datathon_2020.models.predict_model import predict_model
import tempfile
import pandas as pd


def inference(dataset_filepath: str):
    """ Runs modelling scripts to turn preprocessed data from (../processed) into
        to a model (../models)
    """
    logger = logging.getLogger(__name__)
    logger.info(f'learning from file {dataset_filepath}...')

    dataset_filename = Path.cwd().joinpath(dataset_filepath).stem

    make_dataset(dataset_filepath, 'data/interim/')
    encode_dataset('data/interim/' + dataset_filename + '.feather', 'data/processed/', 'models/', False)
    predict_model('data/processed/' + dataset_filename + '_encoded.feather', 'models/model.dump', 'data/predictions/')


def inference_sample(samples):
    logger = logging.getLogger(__name__)
    logger.info(f'performing inference on samples {samples}...')

    preds = dict()
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='wt') as inference_file:
        inputs = pd.DataFrame.from_dict(
            samples,
            orient='index'
        )
        inputs.to_csv(inference_file, index=False)
        inference_file.seek(0)

        inference_file_stem = PurePath(inference_file.name).stem
        inference(inference_file.name)
        preds = pd.read_csv(
            f'./data/predictions/{inference_file_stem}_encoded.csv'
        )[['encounter_id', 'hospital_death']].to_dict(orient='record')
    return preds


@click.command()
@click.argument('dataset_filepath', type=click.Path(exists=True), default='data/raw/training_v2.csv')
def main(dataset_filepath):
    inference(dataset_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
