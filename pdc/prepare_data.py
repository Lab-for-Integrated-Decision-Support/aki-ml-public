import numpy as np
import pandas as pd
from statistics import median

import os
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from set_env_vars import set_all_env_vars
set_all_env_vars()

def load_cohort(data_partition_str):
    cohort = pd.read_parquet(
        os.environ['AKI_DATA_PDC']+'base_cohort_with_labs_and_vitals_v3.parquet',
        filters=[('split_set', '=', data_partition_str)]
    )
    return cohort

def load_and_process_cohort(data_partition_str, impute_method=None, outlier_proportion_to_remove=0.01):
    unprocessed_cohort = load_cohort(data_partition_str)
    
    if impute_method is not None:
        processed_cohort = one_hot_encode_cohort(
            min_max_scale_cohort(
                impute_cohort(
                    unprocessed_cohort, impute_method
                )
            )
        )
    else:
        processed_cohort = one_hot_encode_cohort(
            min_max_scale_cohort(
                unprocessed_cohort
            )
        )
    
    return processed_cohort

"""
Get names of columns with features used as model input.
"""
def get_feature_cols(cohort):
    feature_cols = [
        col for col in cohort.columns if (
            any([agg_str in col for agg_str in ['_mean', '_median', '_min', '_max']])
        ) and (
            all([imp_str not in col for imp_str in ['for_imputation', 'latest_available', 'imputation_median']])
        )
    ] + [
        'is_female', 'ageatadmission', 'bscr', 'is_immunocompromised', 'cardiac_arrest'
    ]
    
    return feature_cols

"""
Impute missing values with choice of impute method. If impute_method='latest',
missing values will be imputed with latest available value if feature is a
lab and there is a previous lab value available and median otherwise. If
impute_method='median', all missing values will be filled with the median
(vital sign medians calculated based on patient's age in years).
"""
def impute_cohort(cohort, impute_method='latest', verbosity=0):
    feature_cols = get_feature_cols(cohort)
    features_to_impute = [col for col in feature_cols if col not in ['is_female', 'ageatadmission', 'is_immunocompromised']]
    
    if 'sex' in cohort.columns:
        cohort['is_female'] = [1 if sex.lower()=='female' else 0 for sex in cohort['sex']]
        cohort.drop('sex', axis=1, inplace=True)

#     # Unify BASE_EXC and BASE_DEF
#     for col in [col for col in cohort.columns if 'BASE_EXC' in col]:
#         cohort[col] = cohort[col].fillna(-1*cohort[col.replace('_EXC', '_DEF')])

    for col in cohort.columns:
        if 'BASE_DEF' in col:
            cohort.drop(col, axis=1, inplace=True)
            
    features_to_impute = [x for x in features_to_impute if 'BASE_DEF' not in x]
    
    if impute_method=='latest':
        # cohort = cohort[~cohort['bscr'].isna()]

        for feature in features_to_impute:
            if cohort[feature].isna().sum() > 0:
                if feature+'_for_imputation' in cohort.columns:
                    cohort[feature].fillna(cohort[feature+'_for_imputation'], inplace=True)
                else:
                    median_impute_val = median(cohort[feature].dropna())
                    cohort[feature] = cohort[feature].fillna(median_impute_val)
                
    elif impute_method=='median':
        impute_df = pd.read_csv(os.environ['AKI_DATA_PDC'] + 'pdc_lab_imputation_values_v2.csv')
        for feature in features_to_impute:
            if verbosity > 0:
                print('Performing median imputation on ' + feature)
            median_impute_col = feature + '_imputation_median'
            if median_impute_col in impute_df.columns:
                cohort[feature] = cohort[feature].fillna(impute_df[median_impute_col].mean())
            else:
                median_impute_val = median(cohort[feature].dropna())
                cohort[feature] = cohort[feature].fillna(median_impute_val)
                
    else:
        raise Exception('Exception: impute_method ' + impute_method + ' not implemented.')
        
    return cohort

def remove_outliers(cohort, proportion_to_exclude=0.01):
    feature_cols = get_feature_cols(cohort)
    features_to_skip = ['ageatadmission', 'is_female', 'sex', 'is_immunocompromised']
    
    for feature in feature_cols:
        if feature not in features_to_skip:
            lower_bound = cohort[feature].dropna().quantile(proportion_to_exclude / 2)
            upper_bound = cohort[feature].dropna().quantile(1 - (proportion_to_exclude / 2))
            cohort[feature] = [x if ((x > lower_bound) & (x < upper_bound)) else np.nan for x in cohort[feature]]
    
    return cohort

def min_max_scale_cohort(cohort):
    # Load min/max values
    with open('pickle/min_feature_val_dict.pickle', 'rb') as infile:
        min_val_dict = pickle.load(infile)
    with open('pickle/max_feature_val_dict.pickle', 'rb') as infile:
        max_val_dict = pickle.load(infile)
    
    # Get feature columns
    feature_cols = get_feature_cols(cohort)
    
    # Scale
    for feature in feature_cols:
        if feature in min_val_dict.keys():
            min_val = min_val_dict[feature]
            max_val = max_val_dict[feature]
            if min_val == max_val:
                cohort[feature] = [0 for _ in range(len(cohort))]
            else:
                cohort[feature] = [(x - min_val) / (max_val - min_val) for x in cohort[feature]]
            
    return cohort

"""
One-hot encode categorical features.
"""
def one_hot_encode_cohort(cohort):
    if 'sex' in cohort.columns:
        cohort['is_female'] = [1 if sex == 'Female' else 0 for sex in cohort['sex']]
        cohort.drop('sex', axis=1, inplace=True)
    
    return cohort
