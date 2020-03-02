import streamlit as st
import wids_datathon_2020 as wids
from wids_datathon_2020.inference import inference_sample
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier
from catboost import Pool
import pickle
# import plotly.figure_factory as ff
import plotly.express as px
import matplotlib.pyplot as plt
import shap
 
st.write('''
    # ISeeYou
    Evaluate the severity of patient's condition in their first 24 hours of intensive ICU care.
''')

model = CatBoostClassifier().load_model(str(Path.cwd().joinpath('models/model.dump')))
inference_cols = ['age', 'ventilated_apache', 'elective_surgery', 'gcs_verbal_apache',  'gcs_motor_apache', 'd1_spo2_min', 'icu_id']

dataset = pd.read_csv(Path.cwd().joinpath('data/raw/training_v2.csv'))
dataset_encoded = pd.read_feather(Path.cwd().joinpath('data/processed/training_v2_train_encoded.feather'))
dataset_encoded = dataset_encoded.append(pd.read_feather(Path.cwd().joinpath('data/processed/training_v2_val_encoded.feather'))).reset_index()
dataset_encoded = dataset_encoded.append(pd.read_feather(Path.cwd().joinpath('data/processed/training_v2_test_encoded.feather'))).reset_index()
dataset_predictions = pd.DataFrame(model.predict_proba(dataset_encoded[model.feature_names_])[:,1], columns=['proba_death'])

dataset_encoded_predictions = pd.concat([dataset_encoded, dataset_predictions], axis=1)

dataset = dataset.add_suffix('_unencoded')
dataset_agg = dataset.merge(dataset_encoded_predictions, left_on='encounter_id_unencoded', right_on='encounter_id')


inference = pd.read_csv(Path.cwd().joinpath('data/raw/unlabeled.csv'))
inference_encoded = pd.read_feather(Path.cwd().joinpath('data/processed/unlabeled_encoded.feather'))


default_sample = dataset_agg[(dataset_agg['hospital_death']==1) & (dataset_agg['proba_death'] > 0.8)].iloc[0,:]

st.sidebar.markdown('Select the Mode of Evaluation:')
mode = st.sidebar.radio('Mode', ['New Patient', 'Historical Patient Data', 'Historical Batch Patient Data'])
st.sidebar.markdown('---')

if 'New Patient' == mode:
    st.sidebar.markdown('Fill in the patient\'s details. For proper estimates it is important to fill out these details comprehensively and accurately:')

    feature_inputs = dict()
    for feature_name in inference_cols:
        feature_inputs[feature_name] = st.sidebar.text_input(f'{feature_name}', default_sample[f'{feature_name}_unencoded'])

    feature_inputs['encounter_id'] = 1001

    preds = pd.DataFrame.from_dict(inference_sample({0: feature_inputs}), orient='columns')
    
    severity_hist = [dataset_agg['proba_death'].sample(1000)]
    group_labels = ['MIT GOSSIS Dataset']

    preds['encounter_id'] = preds['encounter_id'].astype('int')
    preds['hospital_death'] = round(preds['hospital_death'], 2)
    
    st.markdown(f'''## Severity of Patient Condition :pager: ''')
    fig = px.bar(preds, x='encounter_id', y='hospital_death', color='hospital_death', text='hospital_death')

    #fig = ff.create_distplot(severity_hist, group_labels)
    st.plotly_chart(fig)
    
    shap_values = model.get_feature_importance(Pool(dataset_agg[model.feature_names_], label=dataset_agg['hospital_death'], cat_features=dataset_agg[model.feature_names_].iloc[:, model.get_cat_feature_indices()].columns), type='ShapValues')
    expected_values = shap_values[0, -1]
    shap_values = shap_values[:, :-1]
    
    st.write(f'''## Severity Analysis :hospital: ''')
    
    st.markdown(f'''### FACTORS IMPACTING PATIENT HEALTH''')
    st.markdown(f'''
        This plot details the role each of the Patient's details had in determining the severity of their condition.
        The red-coloured segments visualize which variables are acting as stabilizing factors in the Patient's condidtion.
        Blue-coloured segments show the factors that are associated with a deterioration in health.
        The width of these segments represent the proportional influence of a factor as compared to others.
        The bolded model output value represents the severity of the Patient's condition.
    ''')
    shap.force_plot(expected_values, shap_values[1],matplotlib=True, figsize=(20,3), features = dataset_agg[model.feature_names_].columns, link="logit")
    st.pyplot(bbox_inches='tight',dpi=600,pad_inches=0)
    plt.clf()
    #st.image(fig, caption='test', use_column_width=True)
    
    st.markdown(f'''### CUMULATIVE RELATIONSHIPS BETWEEN HEALTH FACTORS''')
    st.markdown(f'''
        This plot visualizes the cumulative effects of a Patient's health-related factors. On the y-axis, is a listing of the notable factors affecting the Patient.
        On the x-axis is a range expressing the severity of a Patient's condition; closer to (1) is worse.
        Having a value of (1) indicates that the Patient is statistically recognized as falling in dire fatal circumstances.
    ''')
    shap.decision_plot(expected_values, shap_values[1], features = dataset_agg[model.feature_names_].columns, link="logit")
    st.pyplot(bbox_inches='tight',dpi=600,pad_inches=0)
    plt.clf()

elif 'Historical Patient Data' == mode:
    st.markdown(f'''
        ## Patient Demographic

        AGE: {default_sample["age_unencoded"]} &nbsp;&nbsp;&nbsp; BMI: {default_sample["bmi_unencoded"]} &nbsp;&nbsp;&nbsp; ELECTIVE_SURGERY: {default_sample["elective_surgery_unencoded"]} <br>
        ETHNICITY: {default_sample["ethnicity_unencoded"]} &nbsp;&nbsp;&nbsp; GENDER: {default_sample["gender_unencoded"]} &nbsp;&nbsp;&nbsp; HEIGHT: {default_sample["height_unencoded"]} <br>
        HOSPITAL_ADMIT_SOURCE: {default_sample["hospital_admit_source_unencoded"]} &nbsp;&nbsp;&nbsp; ICU_ADMIT_SOURCE: {default_sample["icu_admit_source_unencoded"]} &nbsp;&nbsp;&nbsp; ICU_ID: {default_sample["icu_id_unencoded"]} <br>
        ICU_STAY_TYPE: {default_sample["icu_stay_type_unencoded"]} &nbsp;&nbsp;&nbsp; ICU_TYPE: {default_sample["icu_type_unencoded"]} &nbsp;&nbsp;&nbsp; PRE_ICU_LOS_DAYS: {default_sample["pre_icu_los_days_unencoded"]} <br>
        READMISSION_STATUS: {default_sample["readmission_status_unencoded"]} &nbsp;&nbsp;&nbsp; WEIGHT: {default_sample["weight_unencoded"]} <br>
    ''', unsafe_allow_html=True)
    
    st.write(f'''
        ## Severity Estimate
        
    ''')
    st.write(f'''
        ## Apache Comorbidity Results
    ''')
    apache_comorbidity_cols = [f'{x}_unencoded' for x in ['aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis']]
    default_sample[apache_comorbidity_cols]

    st.write(f'''
        ## Vitals
    ''')
    vitals_cols = [f'{x}_unencoded' for x in ['d1_diasbp_invasive_max', 'd1_diasbp_invasive_min',
       'd1_diasbp_max', 'd1_diasbp_min', 'd1_diasbp_noninvasive_max',
       'd1_diasbp_noninvasive_min', 'd1_heartrate_max',
       'd1_heartrate_min', 'd1_mbp_invasive_max', 'd1_mbp_invasive_min',
       'd1_mbp_max', 'd1_mbp_min', 'd1_mbp_noninvasive_max',
       'd1_mbp_noninvasive_min', 'd1_resprate_max', 'd1_resprate_min',
       'd1_spo2_max', 'd1_spo2_min', 'd1_sysbp_invasive_max',
       'd1_sysbp_invasive_min', 'd1_sysbp_max', 'd1_sysbp_min',
       'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min',
       'd1_temp_max', 'd1_temp_min', 'h1_diasbp_invasive_max',
       'h1_diasbp_invasive_min', 'h1_diasbp_max', 'h1_diasbp_min',
       'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min',
       'h1_heartrate_max', 'h1_heartrate_min', 'h1_mbp_invasive_max',
       'h1_mbp_invasive_min', 'h1_mbp_max', 'h1_mbp_min',
       'h1_mbp_noninvasive_max', 'h1_mbp_noninvasive_min',
       'h1_resprate_max', 'h1_resprate_min', 'h1_spo2_max', 'h1_spo2_min',
       'h1_sysbp_invasive_max', 'h1_sysbp_invasive_min', 'h1_sysbp_max',
       'h1_sysbp_min', 'h1_sysbp_noninvasive_max',
       'h1_sysbp_noninvasive_min', 'h1_temp_max', 'h1_temp_min']]
    default_sample[vitals_cols]
elif 'Historical Batch Patient Data' == mode:
    pass

st.write('''
    ## More :heart:
    
    - Find out more about the [App](https://github.com/iainwo/kaggle/tree/master/wids-datathon-2020).
    - Find out more about [WiDS Datathon 2020](https://www.kaggle.com/c/widsdatathon2020/overview).
''')