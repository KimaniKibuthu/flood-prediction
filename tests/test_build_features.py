import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.build_features import (
    FeatureEngineering,
    DataProcessor,
    DataSplitter,
)


@pytest.fixture
def mock_data_processor():
    return MagicMock(spec=DataProcessor)


@pytest.fixture
def mock_data():
    return pd.DataFrame(
        {
            "Rainfall": [1, 2, 3, 4, 5, 6],
            "Longitude": [10, 20, 30, 40, 50, 60],
            "Latitude": [40, 50, 60, 70, 80, 90],
            "Flood": [1, 0, 0, 0, 1, 1],
        }
    )


@pytest.fixture
def mock_reference_data():
    return pd.DataFrame(
        {"Station_Names": ["Station1", "Station2"], "dist_to_water": [100, 200]}
    )


def test_get_spatial_features(mock_data_processor, mock_data):
    mock_data_processor.data = mock_data
    feature_engineering = FeatureEngineering(mock_data_processor)
    feature_engineering.get_spatial_features("Rainfall", "Longitude", "Rain_Longitude")
    assert "Rain_Longitude" in mock_data_processor.data.columns
    assert all(
        mock_data_processor.data["Rain_Longitude"] == [10, 40, 90, 160, 250, 360]
    )


def test_split_data(mock_data_processor, mock_data):
    mock_data_processor.data = mock_data
    data_splitter = DataSplitter(mock_data_processor)
    train, test = data_splitter.split_data(
        mock_data,
        random_state=42,
        test_size=0.2,
        stratify=mock_data["Flood"],
        save=False,
    )
    assert len(train) == 4
    assert len(test) == 2
