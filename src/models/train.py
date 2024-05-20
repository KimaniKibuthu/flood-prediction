import pandas as pd
from typing import Tuple, Dict, Union, List
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
from src.utils import load_config


class DataLoader:
    """
    Class for loading data from files.
    """
    def __init__(self, config: Dict):
        """
        Initialize the DataLoader class.

        Args:
            config (Dict): A dictionary containing the file paths for train and test data.
        """
        self.config = load_config()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the train and test data from the specified file paths.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test data as pandas DataFrames.
        """
        train_data = pd.read_csv(self.config["data"]["train_data_path"])
        test_data = pd.read_csv(self.config["data"]["test_data_path"])
        return train_data, test_data

class ModelTrainer:
    """
    Class for training AutoGluon models.
    """
    def __init__(self, label_column: str, eval_metric: str, presets: List = ['medium_quality', 'optimize_for_deployment']):
        """
        Initialize the ModelTrainer class.

        Args:
            label_column (str): The name of the column containing the target variable.
            eval_metric (str): The evaluation metric to use for model training.
            presets (str, optional): The preset configuration for AutoGluon. Defaults to 'medium_quality'.
        """
        
        self.label_column = label_column
        self.eval_metric = eval_metric
        self.presets = presets

    def train_model(self, train_data: TabularDataset, val_data: TabularDataset) -> TabularPredictor:
        """
        Train an AutoGluon model on the provided data.

        Args:
            train_data (TabularDataset): The training data as a TabularDataset.
            val_data (TabularDataset): The validation data as a TabularDataset.

        Returns:
            TabularPredictor: The trained AutoGluon model.
        """
        predictor = TabularPredictor(
            label=self.label_column,
            eval_metric=self.eval_metric,
            sample_weight="auto_weight",
            path=self.config["modelling"]["models_directory"]
        ).fit(train_data, presets=self.presets)
        return predictor

class ModelEvaluator:
    """
    Class for evaluating trained models.
    """
    def __init__(self, test_data: pd.DataFrame, label_column: str):
        """
        Initialize the ModelEvaluator class.

        Args:
            test_data (pd.DataFrame): The test data as a pandas DataFrame.
            label_column (str): The name of the column containing the target variable.
        """
        self.test_data = test_data
        self.label_column = label_column

    def evaluate_model(self, predictor: TabularPredictor) -> Dict:
        """
        Evaluate a trained model on the test data.

        Args:
            predictor (TabularPredictor): The trained AutoGluon model.

        Returns:
            Dict: A dictionary containing the predictions, true labels, and performance metrics.
        """
        test_no_label = TabularDataset(self.test_data.drop(self.label_column, axis=1))
        y_true = self.test_data[self.label_column].values
        predictions = predictor.predict(test_no_label)
        test = TabularDataset(self.test_data)
        perf = predictor.evaluate(test, auxiliary_metrics=False)
        return {"performance": perf}
        