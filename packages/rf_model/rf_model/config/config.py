import pathlib

import rf_model

import pandas as pd


pd.options.display.max_rows = 10
pd.options.display.max_columns = 10

PACKAGE_ROOT = pathlib.Path(rf_model.__file__).resolve().parent
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'
DATASET_DIR = PACKAGE_ROOT / 'datasets'

# data
TESTING_DATA_FILE = DATASET_DIR / 'test.csv'
TRAINING_DATA_FILE = DATASET_DIR / 'train.csv'
TARGET = 'performance'

# variables
FEATURES = ['gender',
            'father_occupation',
            'family_monthlyIncome',
            'no_members_income',
            'shs_school_type',
            'grade12_cgpa',
            'pr_verbal_language',
            'pr_reading_compre',
            'pr_english',
            'pr_non_verbal',
            'course']

# categorical variables with NA in train set
CATEGORICAL_VARS_WITH_NA = ['father_occupation', 'family_monthlyIncome']

# categorical variables to encode
CATEGORICAL_VARS = ['gender',
                    'father_occupation',
                    'family_monthlyIncome',
                    'no_members_income',
                    'shs_school_type',
                    'course']

# numerical variables with NA in train set
NUMERICAL_VARS_WITH_NA = ['course']

NUMERICAL_NA_NOT_ALLOWED = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_VARS + NUMERICAL_VARS_WITH_NA
]

CATEGORICAL_NA_NOT_ALLOWED = [
    feature for feature in CATEGORICAL_VARS if feature not in CATEGORICAL_VARS_WITH_NA
]

PIPELINE_NAME = "rf_model"
PIPELINE_SAVE_FILE = f"{PIPELINE_NAME}_output_v"

# used for differential testing
ACCEPTABLE_MODEL_DIFFERENCE = 0.05
