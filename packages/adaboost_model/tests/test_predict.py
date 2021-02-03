import math

from adaboost_model.config import config
from adaboost_model.predict import make_prediction
from adaboost_model.processing.data_management import load_dataset


def test_make_single_prediction():
    # Given
    file_name = f"{config.DATASET_DIR}/test.csv"
    test_data = load_dataset(file_name=file_name)
    single_test_json = test_data[0:1].to_json(orient='records')

    print(single_test_json)
    
    # When
    subject = make_prediction(input_data=single_test_json)

    # Then
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], str)
    assert subject.get('predictions')[0] == 'high'


def test_make_multiple_predictions():
    # Given
    file_name = f"{config.DATASET_DIR}/test.csv"
    test_data = load_dataset(file_name=file_name)
    multiple_test_json = test_data.to_json(orient='records')

    # When
    subject = make_prediction(input_data=multiple_test_json)

    # Then
    assert subject is not None
    assert len(subject.get('predictions')) == 760
