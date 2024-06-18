from typing import Dict
import pandas as pd
from datetime import timedelta
from prefect import task, flow
from autogluon.tabular import TabularDataset, TabularPredictor
from src.utils import load_config, logger
from src.train import ModelTrainer, ModelEvaluator
from src.monitoring import check_drift


@task
def condition_check() -> bool:
    # Perform the data drift check
    drift_detected = check_drift()
    return drift_detected


@task
def model_training_task(config: Dict) -> TabularPredictor:
    logger.info("Training and evaluation started...")
    train_dataset = TabularDataset(config["data"]["train_data_path"])

    model_trainer = ModelTrainer(
        label_column=config["modelling"]["target"],
        eval_metric=config["modelling"]["eval_metric"],
        presets=config["modelling"].get(
            "presets", ["medium_quality", "optimize_for_deployment"]
        ),
    )
    predictor = model_trainer.train_model(train_dataset)
    return predictor


@task
def model_evaluation_task(config: Dict) -> Dict:
    predictor = TabularPredictor.load(config["modelling"]["models_directory"])
    test_data = pd.read_csv(config["data"]["test_data_path"])
    model_evaluator = ModelEvaluator(test_data, config["modelling"]["target"])
    performance = model_evaluator.evaluate_model(predictor)
    logger.info(
        f"Model training and evaluation completed with performance metrics: {performance['performance']}"
    )
    return performance


# Load config
config = load_config()

# Define the Prefect Flow
@flow
def full_pipeline_flow(config: Dict):
    # Define tasks
    data_drift_detected = condition_check()
    if data_drift_detected:
        # Run tasks if data drift is detected
        predictor = model_training_task(config)
        performance = model_evaluation_task(predictor, config)

        # Set task dependencies
        predictor >> model_evaluation_task

    else:
        # Do nothing if data drift is not detected
        pass


# Optionally, schedule the flow to run daily
full_pipeline_flow.schedule = timedelta(days=1)

# Run the flow
full_pipeline_flow(config)
