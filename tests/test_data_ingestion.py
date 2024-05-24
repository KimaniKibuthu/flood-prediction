import pytest
import pandas as pd
from src.ingestion import DataIngestion

# Mocking load_config and get_logger functions
def mock_load_config():
    return {"data": {"raw_data_path": "data/raw/FloodPrediction.csv"}}


@pytest.fixture
def data_ingestion(monkeypatch):
    monkeypatch.setattr("src.utils.load_config", mock_load_config)
    return DataIngestion()


def test_load_data_success(monkeypatch, data_ingestion):
    # Mocking pd.read_csv to return a sample dataframe
    def mock_read_csv(file_path):
        return pd.DataFrame({"column1": [1, 2], "column2": [3, 4]})

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    data = data_ingestion.load_data()
    assert not data.empty
    assert list(data.columns) == ["column1", "column2"]


def test_load_data_file_not_found(monkeypatch, data_ingestion):
    # Mocking pd.read_csv to raise FileNotFoundError
    def mock_read_csv(file_path):
        raise FileNotFoundError("File not found")

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    with pytest.raises(FileNotFoundError, match="File not found"):
        data_ingestion.load_data()


def test_load_data_empty_data_error(monkeypatch, data_ingestion):
    # Mocking pd.read_csv to raise EmptyDataError
    def mock_read_csv(file_path):
        raise pd.errors.EmptyDataError("No data")

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    with pytest.raises(pd.errors.EmptyDataError, match="No data"):
        data_ingestion.load_data()


def test_load_data_general_error(monkeypatch, data_ingestion):
    # Mocking pd.read_csv to raise a general Exception
    def mock_read_csv(file_path):
        raise Exception("General error")

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    with pytest.raises(Exception, match="General error"):
        data_ingestion.load_data()
