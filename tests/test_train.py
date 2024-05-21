import pandas as pd
from unittest.mock import patch, MagicMock
from src.train import DataLoader, ModelTrainer, ModelEvaluator
from autogluon.tabular import TabularDataset, TabularPredictor

# Test data
TRAIN_DATA = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "label": [0, 1, 0]})
TEST_DATA = pd.DataFrame({"col1": [4, 5, 6], "col2": [7, 8, 9], "label": [1, 0, 1]})

# Mock configuration
CONFIG = {
    "data": {
        "train_data_path": "path/to/train_data.csv",
        "test_data_path": "path/to/test_data.csv",
    },
    "modelling": {"models_directory": "path/to/models"},
}


@patch("pandas.read_csv")
def test_data_loader(mock_read_csv):
    mock_read_csv.side_effect = [TRAIN_DATA, TEST_DATA]
    data_loader = DataLoader(CONFIG)
    train_data, test_data = data_loader.load_data()
    assert train_data.equals(TRAIN_DATA)
    assert test_data.equals(TEST_DATA)


@patch("autogluon.tabular.TabularPredictor.fit")
def test_model_trainer(mock_fit):
    # Create a mock TabularPredictor instance
    mock_predictor = MagicMock(spec=TabularPredictor)
    mock_fit.return_value = mock_predictor

    train_dataset = TabularDataset(TRAIN_DATA)
    model_trainer = ModelTrainer("label", "f1")

    # Train model
    predictor = model_trainer.train_model(train_dataset)

    # Assert that predictor is indeed the mocked TabularPredictor instance
    assert isinstance(predictor, TabularPredictor)


@patch("autogluon.tabular.TabularPredictor.predict")
@patch("autogluon.tabular.TabularPredictor.evaluate")
def test_model_evaluator(mock_evaluate, mock_predict):
    mock_predictor = TabularPredictor(label="label")
    mock_predict.return_value = [0, 1, 0]
    mock_evaluate.return_value = {"f1": 0.8}
    model_evaluator = ModelEvaluator(TEST_DATA, "label")
    result = model_evaluator.evaluate_model(mock_predictor)
    assert result["performance"] == {"f1": 0.8}
    mock_predict.assert_called_once()
    mock_evaluate.assert_called_once()
