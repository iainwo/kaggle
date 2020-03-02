# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import pickle
import numpy as np
from catboost import CatBoostClassifier
from catboost import Pool
# from sklearn.metrics import confusion_matrix,
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, classification_report
# import pprint
import json
# import shap
# import matplotlib.pyplot as plt


def train_model(train_filepath, val_filepath, test_filepath, output_filepath, report_filepath, figure_filepath):
    """ Runs modelling scripts to turn preprocessed data from (../processed) into
        to a model (../models)
    """
    logger = logging.getLogger(__name__)
    target_col = 'hospital_death'

    logger.info(f'loading train data in {train_filepath}')
    X_train = pd.read_feather(Path.cwd().joinpath(train_filepath))
    y_train = X_train.pop(target_col)

    logger.info(f'loading val data in {val_filepath}')
    X_val = pd.read_feather(Path.cwd().joinpath(val_filepath))
    y_val = X_val.pop(target_col)

    logger.info(f'loading test data in {test_filepath}')
    X_test = pd.read_feather(Path.cwd().joinpath(test_filepath))
    y_test = X_test.pop(target_col)

    exclude_cols = ['apache_4a_hospital_death_prob', 'apache_3j_diagnosis', 'apache_4a_icu_death_prob',
                    'apache_2_diagnosis', 'apache_3j_bodysystem', 'apache_2_bodysystem', 'apache_post_operative']
    include_cols = ['age', 'ventilated_apache', 'elective_surgery',
                    'gcs_verbal_apache',  'gcs_motor_apache', 'd1_spo2_min', 'icu_id']

    categorical_cols = X_train.select_dtypes(include='category').columns
    categorical_cols = [x for x in categorical_cols if x not in exclude_cols and (0 == len(include_cols) or x in include_cols)]
    logger.info(f'using categorical cols: {categorical_cols}')

    numerical_cols = X_train.select_dtypes(include='number').columns
    numerical_cols = [x for x in numerical_cols if x not in exclude_cols and (0 == len(include_cols) or x in include_cols)]
    logger.info(f'using numerical cols: {numerical_cols}')

    X_train = X_train[categorical_cols + numerical_cols]
    X_val = X_val[categorical_cols + numerical_cols]
    X_test = X_test[categorical_cols + numerical_cols]

    logger.info('modelling')
    model_args = {
        # 'custom_loss': ['AUC'],
        'eval_metric': 'AUC',
        'random_seed': 42,
        'logging_level': 'Silent',
        'use_best_model': True
    }

    model = CatBoostClassifier(**model_args)
    model.fit(
        Pool(X_train, y_train, cat_features=categorical_cols),
        eval_set=Pool(X_val, y_val, cat_features=categorical_cols),
        logging_level='Verbose',
        plot=False
    )

    y_val_preds = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    y_val_proba_death = y_val_proba[:, 1]
    val_results = results(y_val, y_val_preds, y_val_proba_death)
    logger.info(f'val results: {val_results}')

    y_test_preds = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)
    y_test_proba_death = y_test_proba[:, 1]
    test_results = results(y_test, y_test_preds, y_test_proba_death)
    logger.info(f'test results: {test_results}')

    logger.info('generating feature importances')
    fi_args = {
        'model': model,
        'cat_cols': categorical_cols,
        'continuous_cols': numerical_cols,
        'figure_filepath': figure_filepath,
        'report_filepath': report_filepath,
        'output_filepath': output_filepath
    }
    fi_args['X'] = X_train
    fi_args['y'] = y_train
    fi_args['file_tag'] = 'train'
    logger.info('generating feature importances - train')
    calc_feature_importances(**fi_args)

    fi_args['X'] = X_val
    fi_args['y'] = y_val
    fi_args['file_tag'] = 'val'
    logger.info('generating feature importances - val')
    calc_feature_importances(**fi_args)

    fi_args['X'] = X_test
    fi_args['y'] = y_test
    fi_args['file_tag'] = 'test'
    logger.info('generating feature importances - test')
    calc_feature_importances(**fi_args)

    logger.info('dumping data and saving model')
    dump_results(Path.cwd().joinpath(report_filepath).joinpath('val-results.txt'), val_results)
    dump_results(Path.cwd().joinpath(report_filepath).joinpath('test-results.txt'), test_results)

    model.save_model(str(Path.cwd().joinpath(output_filepath).joinpath('model.dump')))


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


def calc_feature_importances(
    X,
    y,
    file_tag,
    model,
    cat_cols,
    continuous_cols,
    figure_filepath,
    report_filepath,
    output_filepath
):

    logging.info('computing shap values')
    shap_values = model.get_feature_importance(Pool(X, label=y, cat_features=cat_cols), type='ShapValues')
    # expected_values = shap_values[0, -1]
    shap_values = shap_values[:, :-1]

    # logging.info('computing summary bar plot')
    # fig = shap.summary_plot(shap_values, X, plot_type='bar', max_display=200, show=False)
    # plt.savefig(Path.cwd().joinpath(figure_filepath).joinpath(f'{file_tag}-summary_plot_bar.png'), bbox_inches='tight')

    logging.info('computing feature importances')
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    feature_importance.to_csv(Path.cwd().joinpath(report_filepath).joinpath(
        f'{file_tag}-feature-importances.csv'), index=False)

    # logging.info('computing summary plot')
    # fig = shap.summary_plot(shap_values, X, show=False)
    # plt.savefig(Path.cwd().joinpath(figure_filepath).joinpath(f'{file_tag}-summary_plot.png'), bbox_inches='tight')

    # logging.info('computing summary decision plot')
    # fig = shap.decision_plot(expected_values, shap_values[:2000], feature_names=[x for x in X.columns], show=False)
    # plt.savefig(Path.cwd().joinpath(figure_filepath).joinpath(f'{file_tag}-decision_plot.png'), bbox_inches='tight')

    # fi = model.get_feature_importance(
    #     Pool(X, label=y, cat_features=cat_cols),
    #     type="Interaction"
    # )
    # fi_new = []
    # for k,item in enumerate(fi):
    #     first = X.dtypes.index[fi[k][0]]
    #     second = X.dtypes.index[fi[k][1]]
    #     if first != second:
    #         fi_new.append([first + "_" + second, fi[k][2]])
    # feature_score = pd.DataFrame(fi_new,columns=['Feature-Pair','Score'])
    # feature_score =
    #   feature_score.sort_values(
    #       by='Score',
    #       ascending=False,
    #       inplace=False,
    #       kind='quicksort',
    #       na_position='last'
    #   )
    # ax = feature_score.plot('Feature-Pair', 'Score', kind='bar', color='c')
    # ax.set_title("Pairwise Feature Importance", fontsize = 14)
    # ax.set_xlabel("features Pair")
    # plt.savefig(
    # Path.cwd()
    #   .joinpath(figure_filepath)
    #   .joinpath(f'{file_tag}-pairwise-feature-importances.png'),
    # bbox_inches='tight',
    # figsize=(800, 10))

    # for feature in feature_importance.head(40)['col_name']:
    #     logging.info(f'generating {feature} - feature statistics')
    #     if feature in continuous_cols:
    #         for prediction_type in ['Probability', 'Class']:
    #             model.calc_feature_statistics(
    #                 X,
    #                 y,Í
    #                 feature,
    #                 plot=False,
    #                 prediction_type=prediction_type,
    #                 plot_file=Path.cwd().joinpath(figure_filepath).joinpath(f'{file_tag}-feature-statistics-{feature}-{prediction_type}.html')
    #             )

    # pickle_obj(Path.cwd().joinpath(output_filepath).joinpath(f'{file_tag}-expected-values.pickle'), expected_values)
    # pickle_obj(Path.cwd().joinpath(output_filepath).joinpath(f'{file_tag}-shap-values.pickle'), shap_values)


@click.command()
@click.argument('train_filepath', type=click.Path(exists=True), default='data/processed/training_v2_train_encoded.feather')
@click.argument('val_filepath', type=click.Path(exists=True), default='data/processed/training_v2_val_encoded.feather')
@click.argument('test_filepath', type=click.Path(exists=True), default='data/processed/training_v2_test_encoded.feather')
@click.argument('output_filepath', type=click.Path(exists=True), default='models/')
@click.argument('report_filepath', type=click.Path(exists=True), default='reports/')
@click.argument('figure_filepath', type=click.Path(exists=True), default='reports/figures/')
def main(train_filepath, val_filepath, test_filepath, output_filepath, report_filepath, figure_filepath):
    train_model(train_filepath, val_filepath, test_filepath, output_filepath, report_filepath, figure_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
