# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from wids_datathon_2020.data.make_dataset import make_dataset
from wids_datathon_2020.data.stratify_dataset import stratify_dataset
from wids_datathon_2020.data.encode_dataset import encode_dataset
from wids_datathon_2020.models.train_model import train_model


def learn(dataset_filepath: str):
    """ Runs modelling scripts to turn preprocessed data from (../processed) into
        to a model (../models)
    """
    logger = logging.getLogger(__name__)
    logger.info(f'learning from file {dataset_filepath}...')

    dataset_filename = Path.cwd().joinpath(dataset_filepath).stem

    make_dataset(dataset_filepath, 'data/interim/')
    stratify_dataset('data/interim/' + dataset_filename + '.feather', 'data/interim/')
    encode_dataset('data/interim/' + dataset_filename + '_train.feather', 'data/processed/', 'models/', True)
    encode_dataset('data/interim/' + dataset_filename + '_val.feather', 'data/processed/', 'models/', False)
    encode_dataset('data/interim/' + dataset_filename + '_test.feather', 'data/processed/', 'models/', False)
    train_model(
        'data/processed/' + dataset_filename + '_train_encoded.feather',
        'data/processed/' + dataset_filename + '_val_encoded.feather',
        'data/processed/' + dataset_filename + '_test_encoded.feather',
        'models/',
        'reports/',
        'reports/figures/'
    )


@click.command()
@click.argument('dataset_filepath', type=click.Path(exists=True), default='data/raw/training_v2.csv')
def main(dataset_filepath):
    learn(dataset_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
