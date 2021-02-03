from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from rf_model.processing import preprocessors as pp
from rf_model.config import config

import logging


_logger = logging.getLogger(__name__)


PIPELINE_NAME = "random_forest"

grading_pipe = Pipeline(
    [
        ('categorical_imputer',
            pp.CategoricalImputer(variables=config.CATEGORICAL_VARS_WITH_NA)),

        ('categorical_encoder',
            pp.CategoricalEncoder(variables=config.CATEGORICAL_VARS)),

        ('numerical_inputer',
         pp.NumericalImputer(variables=config.NUMERICAL_VARS_WITH_NA)),

        ('scaler', MinMaxScaler()),
        (PIPELINE_NAME, RandomForestClassifier(random_state=42, n_estimators=30, max_depth=10, min_samples_leaf=9,  max_features=11, class_weight='balanced')
         )
    ]
)
