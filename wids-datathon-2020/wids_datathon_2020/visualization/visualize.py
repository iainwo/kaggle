# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
# import seaborn as sns
import plotly
# import chart_studio.plotly as py
import plotly.figure_factory as ff
# import sys
import missingno as msno
import matplotlib.pyplot as plt


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True), default='data/raw/')
@click.argument('output_filepath', type=click.Path(), default='models/')
@click.argument('report_filepath', type=click.Path(), default='reports/')
@click.argument('figure_filepath', type=click.Path(), default='reports/figures/')
@click.argument('eda_filepath', type=click.Path(), default='reports/eda/')
def main(input_filepath, output_filepath, report_filepath, figure_filepath, eda_filepath):
    """ Runs viz scripts to turn raw data from (../raw) into
        to eda diagrams (../reports/eda)
    """
    logger = logging.getLogger(__name__)

    DATADICT_FILEPATH = Path.cwd().joinpath(input_filepath).joinpath('WiDS Datathon 2020 Dictionary.csv')
    RAW_DF_FILEPATH = Path.cwd().joinpath(input_filepath).joinpath('training_v2.csv')
    INPUT_DIR = Path.cwd().joinpath(input_filepath)
    logger.info(f'modelling with data in {INPUT_DIR.name}')

    logger.info('Load Data')
    datadict = pd.read_csv(DATADICT_FILEPATH)
    train = pd.read_csv(RAW_DF_FILEPATH)

    logger.info('generate types')
    continuous_cols = list(
        list(datadict[datadict['Data Type'] == 'integer']['Variable Name'].unique())
        + list(datadict[datadict['Data Type'] == 'numeric']['Variable Name'].unique())
    )
    # categorical_cols = list(datadict[datadict['Data Type'] == 'string']['Variable Name'].unique())
    # binary_cols = list(datadict[datadict['Data Type'] == 'binary']['Variable Name'].unique())
    # target_cols = 'hospital_death'

    logger.info('generating kde plots')
    for col in continuous_cols:
        try:
            logger.info(f'generating {col} kde')
            plot_data = list()
            plot_labels = list()
            for hospital_id in train['hospital_id'].unique():

                subset = train[train['hospital_id'] == hospital_id]

                tmp = subset[col].fillna(0)
                if 0 < tmp.shape[0]:
                    plot_data.append(tmp)
                    plot_labels.append(str(hospital_id))

            fig = ff.create_distplot(plot_data, plot_labels, show_hist=False, show_rug=False)
            fig.update_layout(title_text=f'{col} curves')
            plotly.offline.plot(fig, filename=str(Path.cwd().joinpath(
                eda_filepath).joinpath('kde/').joinpath(f'kde-{col}.html')), auto_open=False)
        except Exception as e:
            logger.error(e, exc_info=True)

    logger.info('generating missingno figures')
    msno.matrix(train.sample(min(1000, len(train))), figsize=(100, 100), sort='ascending', labels=True)
    plt.savefig(str(Path.cwd().joinpath(eda_filepath).joinpath('missingno/').joinpath(f'matrix-all.png')))

    msno.bar(train.sample(min(1000, len(train))), figsize=(100, 100), labels=True)
    plt.savefig(str(Path.cwd().joinpath(eda_filepath).joinpath('missingno/').joinpath(f'bar-all.png')))

    msno.heatmap(train.sample(min(1000, len(train))), figsize=(100, 100), labels=True)
    plt.savefig(str(Path.cwd().joinpath(eda_filepath).joinpath('missingno/').joinpath(f'heatmap-all.png')))
    # for hospital_id in train['hospital_id'].unique():
    #     plt.close(fig='all')
    #     logger.info(f'plotting msno for {hospital_id}')
    #     subset = train[train['hospital_id'] == hospital_id]

    #     msno.matrix(subset.sample(min(1000, len(subset))), figsize=(100, 100), sort='ascending', labels=True)
    #     plt.savefig(str(Path.cwd().joinpath(eda_filepath).joinpath('missingno/').joinpath(f'matrix-{hospital_id}.png')))

    #     msno.bar(subset.sample(min(1000, len(subset))), figsize=(100, 100), labels=True)
    #     plt.savefig(str(Path.cwd().joinpath(eda_filepath).joinpath('missingno/').joinpath(f'bar-{hospital_id}.png')))

    #     msno.heatmap(subset.sample(min(1000, len(subset))), figsize=(100, 100), labels=True)
    #     plt.savefig(str(Path.cwd().joinpath(eda_filepath).joinpath('missingno/').joinpath(f'heatmap-{hospital_id}.png')))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
