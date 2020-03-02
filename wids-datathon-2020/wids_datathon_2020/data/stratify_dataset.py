# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def stratify_dataset(input_filepath, output_filepath):
    """ Runs data stratifying scripts to form
        train, validation and test sets.
    """
    logger = logging.getLogger(__name__)
    logger.info(f'stratifying file {input_filepath}')

    df = pd.read_feather(Path.cwd().joinpath(input_filepath))

    # Splitting
    strat_cols = ['hospital_death']
    logger.info(f'stratifying on cols: {strat_cols} and hospital_id')

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

    output_filename = Path.cwd().joinpath(input_filepath).stem
    train.reset_index(drop=True).to_feather(Path.cwd().joinpath(output_filepath).joinpath(output_filename + '_train.feather'))
    val.reset_index(drop=True).to_feather(Path.cwd().joinpath(output_filepath).joinpath(output_filename + '_val.feather'))
    test.reset_index(drop=True).to_feather(Path.cwd().joinpath(output_filepath).joinpath(output_filename + '_test.feather'))


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='data/interim/training_v2.feather')
@click.argument('output_filepath', type=click.Path(exists=True), default='data/interim/')
def main(input_filepath, output_filepath):
    stratify_dataset(input_filepath, output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
