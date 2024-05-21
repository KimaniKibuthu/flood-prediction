"""
A module containing the Data Ingestion Pipeline
"""

# Import the necessary modules
import pandas as pd
from typing import Optional, Dict, Any
from src.utils import load_config, logger

# Define Data Ingestion class
class DataIngestion:
    """
    A class to handle data ingestion.
    """
    
    def __init__(self, data_path: Optional[str]=None, config:Optional[Dict[str, Any]] = None):
        """
        Initialize the DataIngestion class.

        Parameters:
        - data_path (str): Path to the data file. If None, it will be loaded from the config.
        - config (dict): Configuration dictionary. If None, it will be loaded using the load_config function.
        """
        self.config = config if config else load_config()
        self.data_path = data_path if data_path else self.config["data"]["raw_data_path"]
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the data from the specified data path.

        Returns:
        - DataFrame: Loaded data as a pandas DataFrame.
        
        Raises:
        - Exception: If there is an error in loading the data.
        """
        try:
            logger.info("Loading data from %s", self.data_path)
            data = pd.read_csv(self.data_path)
            logger.info("Data loaded successfully")
            return data
        except FileNotFoundError as fnf_error:
            logger.error("File not found: %s", self.data_path)
            raise fnf_error
        except pd.errors.EmptyDataError as ede_error:
            logger.error("No data: %s", self.data_path)
            raise ede_error
        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            raise e