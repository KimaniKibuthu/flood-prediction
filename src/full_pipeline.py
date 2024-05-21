"""
Contains the full pipeline
"""

from typing import Dict
from autogluon.tabular import TabularDataset
from src.utils import load_config, logger
from src.ingestion import DataIngestion
from src.cleaning import DataTransformationPipeline
from src.build_features import FeatureEngineeringPipeline, DataProcessor
from src.train import ModelTrainer, ModelEvaluator


def main(config: Dict = load_config()) -> None:
    """
    Run the full data processing, feature engineering, and modelling pipeline.

    Args:
        config (Dict): Configuration dictionary containing paths and settings.
    """
    # Data Transformation
    logger.info("Data Transformation started...")
    transformation_params = {
        "clean_up_column_names": {"columns": {"Flood?": "Flood"}},
        "fill_null_values": {"fill_value": 0, "column_to_fill": "Flood"},
    }
    data_ingestion = DataIngestion(config["data"]["raw_data_path"])
    data_transformation_pipeline = DataTransformationPipeline(
        config=config, data_ingestion=data_ingestion
    )
    data_transformation_pipeline.run_pipeline(transformation_params)
    logger.info("Data Transformation completed successfully")

    # Feature Engineering
    logger.info("Feature Engineering started...")
    data_processor = DataProcessor(
        config["data"]["processed_data_path"], config["data"]["reference_data_path"]
    )
    feature_engineering_pipeline = FeatureEngineeringPipeline(data_processor)
    train_data, test_data = feature_engineering_pipeline.run()
    logger.info("Feature engineering completed successfully")

    # Modelling and Evaluation
    logger.info("Training and evaluation started...")
    train_dataset = TabularDataset(train_data)

    # Train Model
    model_trainer = ModelTrainer(
        label_column=config["modelling"]["target"],
        eval_metric=config["modelling"]["eval_metric"],
        presets=config["modelling"].get(
            "presets", ["medium_quality", "optimize_for_deployment"]
        ),
    )
    predictor = model_trainer.train_model(train_dataset)

    # Evaluate Model
    model_evaluator = ModelEvaluator(test_data, config["modelling"]["target"])
    performance = model_evaluator.evaluate_model(predictor)
    logger.info(
        f"Model training and evaluation completed with performance metrics: {performance['performance']}"
    )


if __name__ == "__main__":
    main()
