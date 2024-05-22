from typing import Dict, Tuple
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from autogluon.tabular import TabularDataset, TabularPredictor
from src.utils import load_config, logger
from src.ingestion import DataIngestion
from src.cleaning import DataTransformationPipeline
from src.build_features import FeatureEngineeringPipeline, DataProcessor
from src.train import ModelTrainer, ModelEvaluator
from datetime import timedelta


def data_transformation_task(config: Dict) -> None:
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


def feature_engineering_task(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Feature Engineering started...")
    data_processor = DataProcessor(
        config["data"]["processed_data_path"], config["data"]["reference_data_path"]
    )
    feature_engineering_pipeline = FeatureEngineeringPipeline(data_processor)
    train_data, test_data = feature_engineering_pipeline.run()
    logger.info("Feature engineering completed successfully")
    return train_data, test_data


def model_training_task(config: Dict, ti) -> TabularPredictor:
    logger.info("Training and evaluation started...")
    test_data = ti.xcom_pull(task_ids="feature_engineering", key="return_value")[1]
    predictor = ti.xcom_pull(task_ids="model_training", key="return_value")
    train_data = ti.xcom_pull(task_ids="feature_engineering", key="return_value")[0]
    train_dataset = TabularDataset(train_data)

    model_trainer = ModelTrainer(
        label_column=config["modelling"]["target"],
        eval_metric=config["modelling"]["eval_metric"],
        presets=config["modelling"].get(
            "presets", ["medium_quality", "optimize_for_deployment"]
        ),
    )
    predictor = model_trainer.train_model(train_dataset)
    return predictor


def model_evaluation_task(
    config: Dict, test_data: pd.DataFrame, predictor: TabularPredictor
) -> None:
    model_evaluator = ModelEvaluator(test_data, config["modelling"]["target"])
    performance = model_evaluator.evaluate_model(predictor)
    logger.info(
        f"Model training and evaluation completed with performance metrics: {performance['performance']}"
    )


default_args = {
    "owner": "flood-prediction-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    "full_pipeline_dag",
    default_args=default_args,
    description="A full pipeline DAG",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
) as dag:

    config = load_config()

    data_transformation = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformation_task,
        op_kwargs={"config": config},
    )

    feature_engineering = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering_task,
        op_kwargs={"config": config},
    )

    model_training = PythonOperator(
        task_id="model_training",
        python_callable=model_training_task,
        op_kwargs={"config": config},
    )

    model_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=model_evaluation_task,
        op_kwargs={"config": config},
    )

    data_transformation >> feature_engineering >> model_training >> model_evaluation
