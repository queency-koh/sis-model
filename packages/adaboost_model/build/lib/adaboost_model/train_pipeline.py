import numpy as np
from sklearn.model_selection import train_test_split

from adaboost_model import pipeline
from adaboost_model.processing.data_management import load_dataset, save_pipeline
from adaboost_model.config import config
from adaboost_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES], data[config.TARGET], test_size=0.1, random_state=0
    )  # we are setting the seed here

    # transform the target
    y_train = y_train.values.ravel()

    pipeline.grading_pipe.fit(X_train[config.FEATURES], y_train)

    _logger.info(f"saving model version: {_version}")
    save_pipeline(pipeline_to_persist=pipeline.grading_pipe)


if __name__ == "__main__":
    run_training()
