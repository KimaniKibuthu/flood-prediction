import os
import json
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    ClassificationQualityMetric,
    DatasetMissingValuesMetric,
    DatasetSummaryMetric,
    ColumnSummaryMetric,
    ColumnDistributionMetric,
)
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from src.utils import load_config, logger
from autogluon.tabular import TabularDataset, TabularPredictor

# Load config
config = load_config()

# Validate config
required_keys = [
    "modelling.models_directory",
    "data.monitoring_data_path",
    "data.train_data_path",
    "data.test_data_path",
]
for key in required_keys:
    keys = key.split(".")
    conf = config
    for k in keys:
        conf = conf.get(k, None)
        if conf is None:
            raise ValueError(f"Missing required config key: {key}")

# Load model
model = TabularPredictor.load(config["modelling"]["models_directory"])


def check_drift():
    try:
        # Construct the path to the 'dataset' folder
        dataset_dir = config["data"]["monitoring_data_path"]

        # Construct the full paths to the CSV files
        reference_path = config["data"]["train_data_path"]
        valid_disturbed_path = config["data"]["test_data_path"]

        if not os.path.exists(reference_path) or not os.path.exists(
            valid_disturbed_path
        ):
            raise FileNotFoundError("Reference or current data file not found")

        # Read the CSV files using the full paths
        reference = TabularDataset(reference_path)
        valid_disturbed = TabularDataset(valid_disturbed_path)
        valid_disturbed["prediction"] = model.predict(valid_disturbed)
        reference["prediction"] = model.predict(reference)
        # set up column mapping
        column_mapping = ColumnMapping()

        # Assuming these columns exist in the dataset
        column_mapping.target = "Flood"
        column_mapping.prediction = "prediction"

        # Dynamically setting up numerical features from the dataset columns
        numerical_features = [
            "Min_Temp",
            "Rainfall",
            "Cloud_Coverage",
            "Station_Number",
            "ALT",
            "rain_latitude",
            "rain_longitude",
            "dist_to_water",
        ]
        column_mapping.numerical_features = [
            col for col in numerical_features if col in reference.columns
        ]

        # List of metrics for exhaustive monitoring
        metrics = [
            DatasetDriftMetric(),
            ColumnDriftMetric("Flood"),
            ClassificationQualityMetric(),
            DatasetMissingValuesMetric(),
            DatasetSummaryMetric(),
            ColumnSummaryMetric(column_name="Rainfall"),
            ColumnDistributionMetric(column_name="Rainfall"),
        ]

        # Add ColumnDriftMetric for each numerical feature
        for feature in column_mapping.numerical_features:
            metrics.append(ColumnDriftMetric(feature))

        # Generate the drift report
        data_drift_report = Report(metrics=metrics)
        data_drift_report.run(
            reference_data=reference,
            current_data=valid_disturbed,
            column_mapping=column_mapping,
        )

        # Extract and print the drift check result
        report_json = json.loads(data_drift_report.json())
        dataset_drift_check = report_json["metrics"][0]["result"][
            "dataset_drift"
        ]  # Adjusted index

        logger.info(f"Dataset drift check result: {dataset_drift_check}")
        return dataset_drift_check

    except Exception as e:
        logger.error(f"An error occurred during drift check: {e}")
        raise


if __name__ == "__main__":
    check_drift()
