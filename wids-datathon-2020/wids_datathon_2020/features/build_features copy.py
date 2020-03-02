# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import bisect
import numpy as np
from itertools import combinations


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../interim) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    IS_USE_COMBOS = False
    IS_DROP_CONFOUNDING_COLS = False

    INPUT_DIR = Path.cwd().joinpath(input_filepath)
    logger.info(f'feature-engineering with data in {INPUT_DIR.name}')

    logger.info('reading envs')
    # cols
    BINARY_COLS = Path.cwd().joinpath(input_filepath).joinpath('binary-cols.pickle')
    CATEGORICAL_COLS = Path.cwd().joinpath(input_filepath).joinpath('categorical-cols.pickle')
    CONTINUOUS_COLS = Path.cwd().joinpath(input_filepath).joinpath('continuous-cols.pickle')
    TARGET_COL = Path.cwd().joinpath(input_filepath).joinpath('target-col.pickle')

    BINARY_COLS_OUT = Path.cwd().joinpath(output_filepath).joinpath('binary-cols.pickle')
    CATEGORICAL_COLS_OUT = Path.cwd().joinpath(output_filepath).joinpath('categorical-cols.pickle')
    CONTINUOUS_COLS_OUT = Path.cwd().joinpath(output_filepath).joinpath('continuous-cols.pickle')
    TARGET_COL_OUT = Path.cwd().joinpath(output_filepath).joinpath('target-col.pickle')

    # data
    TRAIN_CSV = Path.cwd().joinpath(input_filepath).joinpath('train.csv')
    VAL_CSV = Path.cwd().joinpath(input_filepath).joinpath('val.csv')
    TEST_CSV = Path.cwd().joinpath(input_filepath).joinpath('test.csv')
    PREDS_CSV = Path.cwd().joinpath(input_filepath).joinpath('preds.csv')

    TRAIN_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('train.csv')
    VAL_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('val.csv')
    TEST_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('test.csv')
    PREDS_CSV_OUT = Path.cwd().joinpath(output_filepath).joinpath('preds.csv')

    # metadata
    BINARY_ENCODERS = Path.cwd().joinpath(output_filepath).joinpath('binary-encoders.pickle')
    CATEGORICAL_ENCODERS = Path.cwd().joinpath(output_filepath).joinpath('categorical-encoders.pickle')
    TARGET_ENCODERS = Path.cwd().joinpath(output_filepath).joinpath('target-encoders.pickle')
    CONTINUOUS_SCALERS = Path.cwd().joinpath(output_filepath).joinpath('continuous-scalers.pickle')
    CONTINUOUS_FILLERS = Path.cwd().joinpath(output_filepath).joinpath('continuous-fillers.pickle')

    logger.info('reading in data')
    binary_cols = read_obj(BINARY_COLS)
    categorical_cols = read_obj(CATEGORICAL_COLS)
    continuous_cols = read_obj(CONTINUOUS_COLS)
    target_col = read_obj(TARGET_COL)

    model_features = ['apache_4a_hospital_death_prob', 'age', 'ventilated_apache', 'apache_3j_diagnosis', 'd1_spo2_min', 'apache_4a_icu_death_prob', 'd1_heartrate_min', 'd1_lactate_min', 'd1_temp_max', 'bmi', 'd1_wbc_min', 'd1_temp_min', 'd1_resprate_min', 'd1_heartrate_max', 'd1_bun_max', 'd1_sodium_max', 'd1_resprate_max', 'd1_bun_min', 'urineoutput_apache', 'd1_sysbp_noninvasive_min', 'h1_resprate_min', 'd1_platelets_min', 'd1_mbp_min', 'resprate_apache', 'gcs_verbal_apache', 'd1_glucose_min', 'hospital_admit_source', 'd1_pao2fio2ratio_max', 'apache_2_diagnosis', 'wbc_apache', 'pre_icu_los_days', 'd1_arterial_ph_min', 'icu_admit_source', 'd1_sysbp_min', 'gcs_motor_apache', 'd1_arterial_po2_max_na']  # noqa
    # model_cat_features = [x for x in model_features if x in categorical_cols or x in binary_cols]
    model_cat_features = [
        'ventilated_apache',
        'apache_3j_diagnosis',
        'bmi',
        'hospital_admit_source',
        'apache_2_diagnosis',
        'icu_admit_source',
        'apache_2_bodysystem',
        'icu_type',
        'apache_3j_bodysystem',
        'solid_tumor_with_metastasis',
        'elective_surgery',
        'immunosuppression',
        'arf_apache',
        'intubated_apache',
        'cirrhosis'
    ]

    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    test = pd.read_csv(TEST_CSV)
    preds = pd.read_csv(PREDS_CSV)

    logger.info('converting height units to meters')
    train['height'] = train['height']/100
    val['height'] = val['height']/100
    test['height'] = test['height']/100
    preds['height'] = preds['height']/100

    logger.info('recalculating bmi')
    train['bmi'] = train['weight']/(train['height']**2)
    val['bmi'] = val['weight']/(val['height']**2)
    test['bmi'] = test['weight']/(test['height']**2)
    preds['bmi'] = preds['weight']/(preds['height']**2)

    logger.info('creating new feature weightclass')
    train['weightclass'] = train['bmi'].map(weighted_class)
    val['weightclass'] = val['bmi'].map(weighted_class)
    test['weightclass'] = test['bmi'].map(weighted_class)
    preds['weightclass'] = preds['bmi'].map(weighted_class)
    categorical_cols.append('weightclass')

    logger.info('adding pre_icu_los_days_is_bad marker')
    train['pre_icu_los_days_is_bad'] = 0 > train['pre_icu_los_days']
    val['pre_icu_los_days_is_bad'] = 0 > val['pre_icu_los_days']
    test['pre_icu_los_days_is_bad'] = 0 > test['pre_icu_los_days']
    preds['pre_icu_los_days_is_bad'] = 0 > preds['pre_icu_los_days']
    categorical_cols.append('pre_icu_los_days_is_bad')

    logger.info('adding apache_4a_hospital_death_prob_is_bad')
    train['apache_4a_hospital_death_prob_is_bad'] = 0 > train['apache_4a_hospital_death_prob']
    val['apache_4a_hospital_death_prob_is_bad'] = 0 > val['apache_4a_hospital_death_prob']
    test['apache_4a_hospital_death_prob_is_bad'] = 0 > test['apache_4a_hospital_death_prob']
    preds['apache_4a_hospital_death_prob_is_bad'] = 0 > preds['apache_4a_hospital_death_prob']
    categorical_cols.append('apache_4a_hospital_death_prob_is_bad')

    logger.info('adding apache_4a_icu_death_prob_is_bad')
    train['apache_4a_icu_death_prob_is_bad'] = 0 > train['apache_4a_icu_death_prob']
    val['apache_4a_icu_death_prob_is_bad'] = 0 > val['apache_4a_icu_death_prob']
    test['apache_4a_icu_death_prob_is_bad'] = 0 > test['apache_4a_icu_death_prob']
    preds['apache_4a_icu_death_prob_is_bad'] = 0 > preds['apache_4a_icu_death_prob']
    categorical_cols.append('apache_4a_icu_death_prob_is_bad')

    # aggregate icu
    train['hospital_admit_source_is_icu'] = train['hospital_admit_source'].apply(
        lambda x:
        'True' if x in [
            'Other ICU',
            'ICU to SDU',
            'ICU'
        ] else 'False')
    val['hospital_admit_source_is_icu'] = val['hospital_admit_source'].apply(
        lambda x:
        'True' if x in [
            'Other ICU',
            'ICU to SDU',
            'ICU'
        ] else 'False')
    test['hospital_admit_source_is_icu'] = test['hospital_admit_source'].apply(
        lambda x:
        'True' if x in [
            'Other ICU',
            'ICU to SDU',
            'ICU'
        ] else 'False')
    preds['hospital_admit_source_is_icu'] = preds['hospital_admit_source'].apply(
        lambda x:
            'True' if x in [
                'Other ICU',
                'ICU to SDU',
                'ICU'
            ] else 'False')
    categorical_cols.append('hospital_admit_source_is_icu')

    # aggregate ethnicity
    common_cols = [np.nan, 'Other/Unknown']
    train['ethnicity_is_unknown'] = train['ethnicity'].apply(lambda x: True if x in common_cols else False)
    val['ethnicity_is_unknown'] = val['ethnicity'].apply(lambda x: True if x in common_cols else False)
    test['ethnicity_is_unknown'] = test['ethnicity'].apply(lambda x: True if x in common_cols else False)
    preds['ethnicity_is_unknown'] = preds['ethnicity'].apply(lambda x: True if x in common_cols else False)
    categorical_cols.append('ethnicity_is_unknown')

    # aggregate cardiac
    common_cols = ['CTICU', 'CCU-CTICU', 'Cardiac ICU', 'CSICU']
    train['icu_type_is_cardiac'] = train['icu_type'].apply(lambda x: True if x in common_cols else False)
    val['icu_type_is_cardiac'] = val['icu_type'].apply(lambda x: True if x in common_cols else False)
    test['icu_type_is_cardiac'] = test['icu_type'].apply(lambda x: True if x in common_cols else False)
    preds['icu_type_is_cardiac'] = preds['icu_type'].apply(lambda x: True if x in common_cols else False)
    categorical_cols.append('icu_type_is_cardiac')

    # aggregate apache_2_bodysystem
    common_cols = ['Undefined Diagnoses', np.nan, 'Undefined diagnoses']
    train['apache_2_bodysystem_is_undefined'] = train['apache_2_bodysystem'].apply(
        lambda x: True if x in common_cols else False)
    val['apache_2_bodysystem_is_undefined'] = val['apache_2_bodysystem'].apply(lambda x: True if x in common_cols else False)
    test['apache_2_bodysystem_is_undefined'] = test['apache_2_bodysystem'].apply(lambda x: True if x in common_cols else False)
    preds['apache_2_bodysystem_is_undefined'] = preds['apache_2_bodysystem'].apply(
        lambda x: True if x in common_cols else False)
    categorical_cols.append('apache_2_bodysystem_is_undefined')

    if IS_USE_COMBOS:

        logger.info('synthesizing categorical pairs')
        # pair_cols = [
        #     'ethnicity',
        #     'gender',
        #     'hospital_admit_source',
        #     'icu_admit_source',
        #     'icu_stay_type',
        #     'icu_type',
        #     'apache_3j_bodysystem',
        #     'apache_2_bodysystem'
        # ]
        # cmbs = list(combinations(pair_cols, 2))
        cmbs = list(combinations(model_cat_features, 2))
        combo_cols = list()
        for cols in cmbs:
            col_name = f'paired_{"_".join(cols)}'
            combo_cols.append(col_name)
            train[col_name] = concat_columns(train, cols)
            val[col_name] = concat_columns(val, cols)
            test[col_name] = concat_columns(test, cols)
            preds[col_name] = concat_columns(preds, cols)
        categorical_cols.extend(combo_cols)

        logger.info('synthesizing categorical triplets')
        # cmbs = list(combinations(pair_cols, 3))
        cmbs = list(combinations(model_cat_features, 3))
        combo_cols = list()
        for cols in cmbs:
            col_name = f'threed_{"_".join(cols)}'
            combo_cols.append(col_name)
            train[col_name] = concat_columns(train, cols)
            val[col_name] = concat_columns(val, cols)
            test[col_name] = concat_columns(test, cols)
            preds[col_name] = concat_columns(preds, cols)
        categorical_cols.extend(combo_cols)

        # logger.info('synthesizing binary pairs')
        # combo_cols = list()
        # for cols in list(combinations(binary_cols, 2)):
        #     col_name = f'paired_binary_{"_".join(cols)}'
        #     combo_cols.append(col_name)
        #     train[col_name] = concat_columns(train, cols)
        #     val[col_name] = concat_columns(val, cols)
        #     test[col_name] = concat_columns(test, cols)
        #     preds[col_name] = concat_columns(preds, cols)
        # categorical_cols.extend(combo_cols)

    logger.info('typifying...')
    # continuous
    train[continuous_cols] = train[continuous_cols].astype('float32')
    val[continuous_cols] = val[continuous_cols].astype('float32')
    test[continuous_cols] = test[continuous_cols].astype('float32')
    preds[continuous_cols] = preds[continuous_cols].astype('float32')

    # cats
    train[categorical_cols] = train[categorical_cols].astype('str').astype('category')
    val[categorical_cols] = val[categorical_cols].astype('str').astype('category')
    test[categorical_cols] = test[categorical_cols].astype('str').astype('category')
    preds[categorical_cols] = preds[categorical_cols].astype('str').astype('category')

    # binary
    train[binary_cols] = train[binary_cols].astype('str').astype('category')
    val[binary_cols] = val[binary_cols].astype('str').astype('category')
    test[binary_cols] = test[binary_cols].astype('str').astype('category')
    preds[binary_cols] = preds[binary_cols].astype('str').astype('category')

    # targets
    train[target_col] = train[target_col].astype('str').astype('category')
    val[target_col] = val[target_col].astype('str').astype('category')
    test[target_col] = test[target_col].astype('str').astype('category')

    logger.info("dropping")
    train = train.dropna(how='all')
    val = val.dropna(how='all')
    test = test.dropna(how='all')
    preds = preds.dropna(how='all')

    if IS_DROP_CONFOUNDING_COLS:
        logger.info('dropping confounding features')
        confounding_cols = ['apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob']

        # remove confounding vars - biases and undue variance
        for x in confounding_cols:
            continuous_cols.remove(x)
        train = train.drop(columns=confounding_cols)
        val = val.drop(columns=confounding_cols)
        test = test.drop(columns=confounding_cols)
        preds = preds.drop(columns=confounding_cols)

    train = train[[target_col] + continuous_cols + categorical_cols + binary_cols]
    val = val[[target_col] + continuous_cols + categorical_cols + binary_cols]
    test = test[[target_col] + continuous_cols + categorical_cols + binary_cols]
    preds = preds[continuous_cols + categorical_cols + binary_cols]

    # keep = ['encounter_id']
    # model_features = ['apache_4a_hospital_death_prob', 'age', 'apache_3j_diagnosis', 'd1_heartrate_min', 'bmi', 'apache_4a_icu_death_prob', 'd1_spo2_min', 'd1_bun_min', 'ventilated_apache', 'd1_temp_min', 'paired_ventilated_apache_icu_admit_source', 'd1_lactate_min', 'd1_bun_max', 'd1_platelets_min', 'd1_wbc_min', 'd1_sodium_max', 'd1_resprate_min', 'd1_resprate_max', 'urineoutput_apache', 'd1_arterial_ph_min', 'd1_temp_max', 'd1_mbp_min', 'apache_2_diagnosis', 'd1_sysbp_noninvasive_min', 'd1_heartrate_max', 'd1_sysbp_min', 'gcs_verbal_apache', 'resprate_apache', 'd1_pao2fio2ratio_max', 'd1_glucose_min', 'gcs_motor_apache', 'pre_icu_los_days', 'wbc_apache_na', 'h1_resprate_min', 'hospital_admit_source', 'wbc_apache', 'icu_admit_source', 'd1_arterial_ph_min_na', 'urineoutput_apache_na', 'paired_ventilated_apache_hospital_admit_source'] # noqa
    # model_features = ['apache_4a_hospital_death_prob', 'age', 'apache_3j_diagnosis', 'ventilated_apache', 'd1_spo2_min', 'apache_4a_icu_death_prob', 'd1_heartrate_min', 'd1_bun_min', 'bmi', 'd1_wbc_min', 'd1_heartrate_max', 'd1_resprate_max', 'd1_temp_max', 'd1_temp_min', 'd1_resprate_min', 'd1_sodium_max', 'gcs_verbal_apache', 'd1_lactate_min', 'd1_bun_max', 'd1_glucose_min', 'h1_resprate_min', 'd1_platelets_min', 'd1_mbp_min', 'hospital_admit_source', 'h1_heartrate_max', 'd1_arterial_ph_min', 'apache_3j_bodysystem', 'gcs_eyes_apache', 'h1_sysbp_min', 'bun_apache', 'apache_2_diagnosis', 'resprate_apache', 'd1_sysbp_min', 'pre_icu_los_days', 'd1_glucose_max', 'd1_creatinine_max', 'd1_arterial_pco2_max_na', 'd1_hco3_max', 'd1_hemaglobin_max', 'urineoutput_apache_na', 'd1_pao2fio2ratio_max', 'icu_type', 'wbc_apache', 'd1_sysbp_noninvasive_min', 'h1_diasbp_max', 'd1_lactate_max', 'gcs_motor_apache', 'd1_mbp_noninvasive_min', 'icu_admit_source', 'd1_arterial_pco2_max', 'd1_arterial_ph_max', 'map_apache', 'h1_mbp_noninvasive_min', 'bilirubin_apache', 'urineoutput_apache', 'h1_heartrate_min', 'd1_bilirubin_min', 'd1_albumin_max', 'd1_sysbp_noninvasive_max', 'h1_temp_max', 'height', 'd1_potassium_min', 'heart_rate_apache', 'd1_hemaglobin_min', 'd1_sysbp_max', 'd1_albumin_min', 'weight', 'glucose_apache', 'paco2_apache_na', 'd1_pao2fio2ratio_min', 'h1_temp_min', 'd1_platelets_max', 'age_na', 'd1_arterial_pco2_min', 'h1_resprate_max', 'd1_calcium_max', 'elective_surgery', 'd1_hematocrit_min', 'temp_apache', 'd1_spo2_max', 'weightclass', 'h1_mbp_min', 'd1_creatinine_min', 'solid_tumor_with_metastasis', 'd1_arterial_po2_min', 'd1_hco3_min', 'paco2_apache', 'd1_hematocrit_max', 'd1_calcium_min', 'd1_inr_min', 'd1_wbc_max', 'hematocrit_apache', 'h1_sysbp_invasive_max', 'sodium_apache', 'd1_inr_max', 'h1_spo2_min', 'h1_diasbp_min', 'h1_inr_max', 'h1_sysbp_noninvasive_max', 'd1_bilirubin_max', 'd1_sysbp_invasive_min', 'h1_diasbp_noninvasive_min', 'd1_diasbp_min', 'apache_4a_hospital_death_prob_is_bad', 'd1_arterial_po2_max', 'creatinine_apache', 'd1_diasbp_invasive_min', 'd1_potassium_max_na', 'pao2_apache', 'h1_glucose_min', 'creatinine_apache_na', 'ph_apache', 'd1_mbp_invasive_min', 'h1_diasbp_noninvasive_max', 'd1_pao2fio2ratio_min_na', 'h1_wbc_max', 'h1_sysbp_max', 'glucose_apache_na', 'h1_mbp_noninvasive_max', 'd1_sodium_min'] # noqa
    # categorical_cols = [x for x in categorical_cols if x in model_features or x in keep]
    # binary_cols = [x for x in binary_cols if x in model_features or x in keep]
    # continuous_cols = [x for x in continuous_cols if x in model_features or x in keep]

    logger.info('dropping identifiers from list cols')
    drop_cols = ['encounter_id', 'hospital_id', 'patient_id', 'icu_id']  # , 'ethnicity', 'readmission_status']
    for col in drop_cols:
        if col in continuous_cols:
            continuous_cols.remove(col)
        if col in categorical_cols:
            categorical_cols.remove(col)
        if col in binary_cols:
            binary_cols.remove(col)

    logger.info('filling')
    train, fillers = fill(train, continuous_cols)
    val, _ = fill(val, continuous_cols, fillers)
    test, _ = fill(test, continuous_cols, fillers)
    preds, _ = fill(preds, continuous_cols, fillers)
    # train, fillers = fill_by_cluster(train, continuous_cols)
    # val, _ = fill_by_cluster(val, continuous_cols, fillers)
    # test, _ = fill_by_cluster(test, continuous_cols, fillers)
    # preds, _ = fill_by_cluster(preds, continuous_cols, fillers)
    categorical_cols.extend([f'{x}_na' for x in continuous_cols])

    logger.info('normalizing')
    train, scalers = normalize(train, continuous_cols)
    val, _ = normalize(val, continuous_cols, scalers)
    test, _ = normalize(test, continuous_cols, scalers)
    preds, _ = normalize(preds, continuous_cols, scalers)

    logger.info('label encoding categoricals')

    # to handle converting fill_nas from bool to str
    train[categorical_cols] = train[categorical_cols].astype('str').astype('category')
    val[categorical_cols] = val[categorical_cols].astype('str').astype('category')
    test[categorical_cols] = test[categorical_cols].astype('str').astype('category')
    preds[categorical_cols] = preds[categorical_cols].astype('str').astype('category')

    train, label_encoders = labelencode(train, categorical_cols)
    val, _ = labelencode(val, categorical_cols, label_encoders)
    test, _ = labelencode(test, categorical_cols, label_encoders)
    preds, _ = labelencode(preds, categorical_cols, label_encoders)

    logger.info('label encoding targets')
    train, target_encoders = labelencode(train, [target_col])
    val, _ = labelencode(val, [target_col], target_encoders)
    test, _ = labelencode(test, [target_col], target_encoders)

    logger.info('one hot encoding categoricals')
    # TODO: change to ohe later
    train, ohe_encoders = labelencode(train, binary_cols)
    val, _ = labelencode(val, binary_cols, ohe_encoders)
    test, _ = labelencode(test, binary_cols, ohe_encoders)
    preds, _ = labelencode(preds, binary_cols, ohe_encoders)

    logger.info('persisting data')
    # cols
    pickle_obj(BINARY_COLS_OUT, binary_cols)
    pickle_obj(CATEGORICAL_COLS_OUT, categorical_cols)
    pickle_obj(CONTINUOUS_COLS_OUT, continuous_cols)
    pickle_obj(TARGET_COL_OUT, target_col)

    # metadata
    pickle_obj(BINARY_ENCODERS, ohe_encoders)
    pickle_obj(CATEGORICAL_ENCODERS, label_encoders)
    pickle_obj(TARGET_ENCODERS, target_encoders)
    pickle_obj(CONTINUOUS_SCALERS, scalers)
    pickle_obj(CONTINUOUS_FILLERS, fillers)

    # data
    train.to_csv(TRAIN_CSV_OUT, index=False)
    val.to_csv(VAL_CSV_OUT, index=False)
    test.to_csv(TEST_CSV_OUT, index=False)
    preds.to_csv(PREDS_CSV_OUT, index=False)


def read_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    return None


def concat_columns(df, columns):
    value = df[columns[0]].astype(str) + ' '
    for col in columns[1:]:
        value += df[col].astype(str) + ' '
    return value


def weighted_class(x):
    if pd.isna(x):
        return np.nan
    elif x < 15:
        return 'very severely underweight'
    elif x >= 15 and x < 16:
        return 'severely weight'
    elif x >= 16 and x < 18.5:
        return 'underweight'
    elif x >= 18.5 and x < 25:
        return 'healthy weight'
    elif x >= 25 and x < 30:
        return 'overweight'
    elif x >= 30 and x < 35:
        return 'class 1'
    elif x >= 35 and x < 40:
        return 'class 2'
    else:
        return 'class 3'
# def fill(df, cols, fillers=None):
#     if None is fillers:
#         fillers = dict()

#         for col in cols:
#             logging.info(f'generating fill col {col}')
#             for hospital_id in df['hospital_id'].unique():

#                 subset = df[df['hospital_id'] == hospital_id]
#                 hospital_col_key = f'{col}_{hospital_id}'

#                 if hospital_col_key not in fillers:
#                     fillers[hospital_col_key] = subset[col].dropna().median()

#             fillers[col] = df[col].dropna().median()

#     for col in cols:
#         logging.info(f'fillling {col}')
#         for hospital_id in df['hospital_id'].unique():
#             subset = df[df['hospital_id'] == hospital_id]
#             hospital_col_key = f'{col}_{hospital_id}'

#             if hospital_col_key in fillers:
#                 df.loc[df['hospital_id'] == hospital_id, col] = subset[col].fillna(fillers[hospital_col_key])
#             else:
#                 df.loc[df['hospital_id'] == hospital_id, col] = subset[col].fillna(fillers[col])

#             df.loc[df['hospital_id'] == hospital_id, f'{col}_na'] = pd.isnull(subset[col])

#     return df, fillers


def fill_by_cluster(df, cols, fillers=None):
    if None is fillers:
        fillers = dict()

        for col in cols:
            logging.info(f'generating fill col {col}')
            for hospital_id in df['cluster_id'].unique():

                subset = df[df['cluster_id'] == hospital_id]
                hospital_col_key = f'{col}_{hospital_id}'

                if hospital_col_key not in fillers:
                    fillers[hospital_col_key] = subset[col].dropna().mean()

            fillers[col] = df[col].dropna().mean()

    for col in cols:
        logging.info(f'fillling {col}')
        for hospital_id in df['cluster_id'].unique():
            subset = df[df['cluster_id'] == hospital_id]
            hospital_col_key = f'{col}_{hospital_id}'

            if hospital_col_key in fillers:
                df.loc[df['cluster_id'] == hospital_id, col] = subset[col].fillna(fillers[hospital_col_key])
            else:
                df.loc[df['cluster_id'] == hospital_id, col] = subset[col].fillna(fillers[col])

            df.loc[df['cluster_id'] == hospital_id, f'{col}_na'] = pd.isnull(subset[col])

    return df, fillers


def fill(df, cols, fillers=None, is_demarc=True, fill_val=None):
    if None is fillers:
        fillers = dict()
    for col in cols:
        if col not in fillers:
            if fill_val is not None:
                fillers[col] = -999
            else:
                fillers[col] = df[col].dropna().mean()

        if is_demarc:
            df[f'{col}_na'] = pd.isnull(df[col])
        df[col] = df[col].fillna(fillers[col])

    return df, fillers


def normalize(df, cols, scalers=None):
    if None is scalers:
        scalers = dict()

    for col in cols:
        if col not in scalers:
            scalers[col] = StandardScaler(with_mean=True, with_std=True)
            scalers[col].fit(df[col].values.reshape(-1, 1))

        scaler = scalers[col]
        df[col] = scaler.transform(df[col].values.reshape(-1, 1))
    return df, scalers


def labelencode(df, cols, encoders=None, unknown_value='UNK'):
    if None is encoders:
        encoders = dict()

    for col in cols:
        if col not in encoders:
            le = LabelEncoder()
            le.fit(df[col].values)

            # add unknown val to cats
            cats = le.classes_.tolist()
            bisect.insort_left(cats, unknown_value)

            # redefine cats on le
            le.classes_ = np.asarray(cats)

            encoders[col] = le

        le = encoders[col]
        df[col] = df[col].map(lambda x: unknown_value if x not in le.classes_ else x)
        df[col] = le.transform(df[col].values)

    return df, encoders


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
