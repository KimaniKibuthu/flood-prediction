""" 
A module on building features
"""

import pandas as pd
from src.utils import load_config
from sklearn.model_selection import train_test_split
from typing import Tuple


# Create feature engineering class
class DataProcessor:
    def __init__(self, data_path: str, reference_data_path: str) -> None:
        """
        Initialize the DataProcessor class.

        Parameters:
        - data_path (str): Path to the data file.
        - reference_data_path (str): Path to the reference data file.
        """
        self.data = pd.read_csv(data_path)
        self.reference_data = pd.read_csv(reference_data_path)


class FeatureEngineering:
    def __init__(self, data_processor: DataProcessor) -> None:
        """
        Initialize the FeatureEngineering class.

        Parameters:
        - data_processor (DataProcessor): An instance of DataProcessor.
        """
        self.data_processor = data_processor

    def get_spatial_features(
        self, column1: str, column2: str, new_column_name: str
    ) -> None:
        """
        Generate spatial features based on existing columns.

        Parameters:
        - column1 (str): Name of the first column.
        - column2 (str): Name of the second column.
        - new_column_name (str): Name for the new column.
        """
        self.data_processor.data[new_column_name] = (
            self.data_processor.data[column1] * self.data_processor.data[column2]
        )


class DataSplitter:
    def __init__(self, data_processor: DataProcessor) -> None:
        """
        Initialize the DataSplitter class.

        Parameters:
        - data_processor (DataProcessor): An instance of DataProcessor.
        """
        self.data_processor = data_processor
        self.config = load_config()

    def split_data(
        self,
        data: pd.DataFrame,
        random_state: int,
        test_size: float,
        stratify: pd.Series,
        save: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the data.
        - random_state (int): Random seed for reproducibility.
        - test_size (float): Proportion of the dataset to include in the test split.
        - stratify (pd.Series): Series containing the target variable to stratify by.
        - save (bool): Whether to save the split data to CSV files.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing train and test DataFrames.
        """
        train, test = train_test_split(
            data, test_size=test_size, stratify=stratify, random_state=random_state
        )
        if save:
            train.to_csv(self.config["data"]["train_data_path"], index=False)
            test.to_csv(self.config["data"]["test_data_path"], index=False)
        return train, test


class FeatureEngineeringPipeline(FeatureEngineering, DataSplitter):
    def __init__(self, data_processor: DataProcessor) -> None:
        """
        Initialize the FeatureEngineeringPipeline class.

        Parameters:
        - data_processor (DataProcessor): An instance of DataProcessor.
        """
        super().__init__(data_processor)
        self.config = load_config()

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the feature engineering pipeline.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing train and test DataFrames.
        """
        self.get_spatial_features("Rainfall", "LONGITUDE", "rain_longitude")
        self.get_spatial_features("Rainfall", "LATITUDE", "rain_latitude")

        temp_data = self.data_processor.reference_data[
            ["Station_Names", "dist_to_water"]
        ]
        full_data = pd.merge(
            self.data_processor.data, temp_data, on="Station_Names", how="inner"
        )
        features_set = self.config["data"]["features_set"]
        full_data = full_data[features_set]
        train, test = self.split_data(
            full_data,
            random_state=self.config["modelling"]["random_state"],
            test_size=self.config["modelling"]["test_size"],
            stratify=full_data["Flood"],
            save=True,
        )
        return train, test
