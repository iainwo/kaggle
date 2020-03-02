# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import zipfile


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='data/external/widsdatathon2020.zip')
@click.argument('output_filepath', type=click.Path(exists=True), default='data/raw/')
def unzip_dataset(input_filepath, output_filepath):
    """ Unzip a dataset from external data (data/external) into
        the raw staging directory (data/raw).
    """
    logger = logging.getLogger(__name__)
    logger.info(f'unzipping file {input_filepath} to output filepath {output_filepath}')

    DATASET = Path.cwd().joinpath(input_filepath)

    with zipfile.ZipFile(DATASET, 'r') as zip_ref:
        zip_ref.extractall(output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    unzip_dataset()
