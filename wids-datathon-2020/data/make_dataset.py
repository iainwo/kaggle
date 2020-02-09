# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import zipfile
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
@click.argument('external_filepath', type=click.Path(exists=True), default='data/external/')
def main(input_filepath, output_filepath, external_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    INPUT_FILEPATH = Path.cwd().joinpath(input_filepath)
    DATASET = Path.cwd().joinpath(external_filepath).joinpath('widsdatathon2020.zip')
    
    DATADICT_FILEPATH = Path.cwd().joinpath(input_filepath).joinpath('WiDS Datathon 2020 Dictionary.csv')
    DATA = Path.cwd().joinpath(input_filepath).joinpath('training_v2.csv')

    with zipfile.ZipFile(DATASET, 'r') as zip_ref:
        zip_ref.extractall(INPUT_FILEPATH)
        
    datadict = pd.read_csv(DATADICT_FILEPATH)
    continuous_cols = list(
        list(datadict[datadict['Data Type'] == 'integer']['Variable Name'].unique())
        + list(datadict[datadict['Data Type'] == 'numeric']['Variable Name'].unique())
    )
    categorical_cols = list(datadict[datadict['Data Type'] == 'string']['Variable Name'].unique())
    binary_cols = list(datadict[datadict['Data Type'] == 'binary']['Variable Name'].unique())
    
    logger.info(f'Continuous cols: {continuous_cols}')
    logger.info(f'Categorical cols: {categorical_cols}')
    logger.info(f'Binary cols: {binary_cols}')
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
