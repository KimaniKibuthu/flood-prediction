import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.data.cleaning import DataTransformation
from src.data.ingestion import DataIngestion

# Mocking load_config function
def mock_load_config() -> dict:
    return {
        "data": {
            "raw_data_path": "test_data.csv",
            "processed_data_path": "processed_data.csv"
        }
    }

# Mocking DataIngestion class
class MockDataIngestion:
    def load_data(self) -> pd.DataFrame:
        return pd.DataFrame({
            "old_column1": [1, 2, None],
            "old_column2": [3, None, 5]
        })

@pytest.fixture
def data_transformation(monkeypatch) -> DataTransformation:
    monkeypatch.setattr("src.utils.load_config", mock_load_config)
    return DataTransformation(config=mock_load_config(), data_ingestion=MockDataIngestion())

def test_clean_up_column_names(data_transformation: DataTransformation) -> None:
    columns_to_rename = {"old_column1": "new_column1", "old_column2": "new_column2"}
    result = data_transformation.clean_up_column_names(columns_to_rename)
    assert list(result.columns) == ["new_column1", "new_column2"]

def test_fill_null_values(data_transformation: DataTransformation) -> None:
    fill_value = 0
    column_to_fill = "old_column1"
    result = data_transformation.fill_null_values(fill_value, column_to_fill)
    assert result["old_column1"].isnull().sum() == 0
    assert (result["old_column1"] == 0).sum() == 1

@patch("pandas.DataFrame.to_csv")
def test_save_transformed_data(mock_to_csv: MagicMock, data_transformation: DataTransformation) -> None:
    data_transformation.save_transformed_data()
    mock_to_csv.assert_called_once_with("processed_data.csv", index=False)
