from evidently import ColumnMapping
from evidently.model_monitoring import ModelMonitor
from evidently.model_monitoring import DataDriftMonitor
from evidently.tests import (
    TestNumbersDistributionCalibration,
    TestDataDistribution,
)
from src.utils import load_config


config = load_config()


def check_model_drift(reference_data, current_data, config):
    """
    Check for model drift using Evidently.

    Args:
        reference_data (pd.DataFrame): Reference data for model monitoring.
        current_data (pd.DataFrame): Current data for model monitoring.
        model: Trained model for prediction.

    Returns:
        bool: True if model drift is detected, False otherwise.
    """
    column_mapping = ColumnMapping()
    column_mapping.target = config["modelling"]["target"]
    column_mapping.prediction = "prediction"
    column_mapping.numerical_features = config["modelling"]["features_set"]

    monitor = ModelMonitor(
        column_mapping=column_mapping,
        categorical_target=column_mapping.target,  # Replace with your target column name
    )

    # Add data drift monitors
    monitor.add_data_drift_monitor(
        DataDriftMonitor(
            TestDataDistribution(
                column_mapping.numerical_features,
                quantile_range=(0.05, 0.95),
            ),
            TestNumbersDistributionCalibration(
                column_mapping.categorical_features,
            ),
        )
    )

    # Run Evidently model monitoring
    monitor.monitor(
        reference_data=reference_data,
        current_data=current_data,
        prediction=model.predict(current_data),
    )

    # Check if model drift is detected
    if monitor.has_drifted():
        return True
    else:
        return False
