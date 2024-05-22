""" 
A module containg the Data Transformation steps
"""

# import the necessary libraries
import pandas as pd
from typing import Dict, Optional, Any
from src.utils import logger, load_config
from src.ingestion import DataIngestion


class DataTransformation:
    """
    A class to handle data transformation.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_ingestion: Optional[DataIngestion] = None,
    ):
        """
        Initialize the DataTransformation class.

        Parameters:
        - config (Optional[Dict[str, Any]]): Configuration dictionary. If None, it will be loaded using the load_config function.
        - data_ingestion (Optional[DataIngestion]): An instance of DataIngestion. If None, a new instance will be created.
        """
        self.config = config if config else load_config()
        self.data = (
            data_ingestion.load_data()
            if data_ingestion
            else DataIngestion().load_data()
        )
        logger.info(
            "DataTransformation initialized with data from %s",
            self.config["data"]["raw_data_path"],
        )

    def clean_up_column_names(self, columns: Dict[str, str]) -> pd.DataFrame:
        """
        Clean up column names by renaming them.

        Parameters:
        - columns (Dict[str, str]): A dictionary mapping old column names to new column names.

        Returns:
        - pd.DataFrame: DataFrame with renamed columns.
        """
        logger.info("Cleaning up column names: %s", columns)
        self.data.rename(columns=columns, inplace=True)
        logger.info("Column names cleaned: %s", self.data.columns)
        return self.data

    def fill_null_values(self, fill_value: Any, column_to_fill: str) -> pd.DataFrame:
        """
        Fill null values in the specified column.

        Parameters:
        - fill_value (Any): The value to fill null values with.
        - column_to_fill (str): The name of the column to fill null values in.

        Returns:
        - pd.DataFrame: DataFrame with filled null values.
        """
        logger.info(
            "Filling null values in column '%s' with value '%s'",
            column_to_fill,
            fill_value,
        )
        self.data[column_to_fill] = self.data[column_to_fill].fillna(fill_value)
        logger.info("Null values filled in column '%s'", column_to_fill)
        return self.data

    def save_transformed_data(self) -> None:
        """
        Save the transformed data to a CSV file.
        """
        logger.info(
            "Saving transformed data to %s", self.config["data"]["processed_data_path"]
        )
        self.data.to_csv(self.config["data"]["processed_data_path"], index=False)
        logger.info(
            "Transformed data saved to %s", self.config["data"]["processed_data_path"]
        )


class DataTransformationPipeline(DataTransformation):
    """
    A class to handle a data transformation pipeline.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        data_ingestion: Optional[DataIngestion] = None,
    ):
        """
        Initialize the DataTransformationPipeline class.

        Parameters:
        - config (Optional[Dict[str, Any]]): Configuration dictionary. If None, it will be loaded using the load_config function.
        - data_ingestion (Optional[DataIngestion]): An instance of DataIngestion. If None, a new instance will be created.
        """
        super().__init__(config, data_ingestion)

    def run_pipeline(self, transformation_params: Dict[str, Dict[str, Any]]) -> None:
        """
        Run all data transformation methods defined in DataTransformation with given parameters.

        Parameters:
        - transformation_params (Dict[str, Dict[str, Any]]): A dictionary where keys are method names and values are dictionaries of parameters for each method.
        """
        for method_name, params in transformation_params.items():
            transformation_method = getattr(self, method_name)
            transformation_method(**params)
