import pandas as pd
import joblib
import numpy as np
import mlflow
import sagemaker
import tempfile
import xgboost as xgb
from sklearn.pipeline import Pipeline


project = "shipping-days-prediction"
sagemaker_session = sagemaker.Session()

s3_bucket = sagemaker_session.default_bucket()
bucket_prefix = "shipping_days_predicition"
query_output_s3_path = (
    f"s3://{s3_bucket}/athena_queries/shipping-days-prediction-dataset"
)

experiment_name = f"{project}-experiment-20-19-02-38"
output_s3_prefix = f"s3://{s3_bucket}/{bucket_prefix}"
tracking_server_arn = (
    f"arn:aws:sagemaker:eu-west-1:602275515347:mlflow-tracking-server/{project}"
)

features_to_remove = [
    "record_id",
    "expected_shipping_days",
    "order_date",
    "event_time",
    "x_shipping_distance",
    "y_shipping_distance",
    "write_time",
    "api_invocation_time",
    "is_deleted",
]
categorical_features_names = [
    "carrier",
    "in_bulk_order",
    "shipping_origin",
    "on_time_delivery",
    "computer_brand",
    "shipping_priority",
    "computer_model",
]
target = "actual_shipping_days"


def run(run_id: str, train_data, validation_data):
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name=experiment_name)

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="ModelTraining", nested=True) as training_run:
            training_run_id = training_run.info.run_id

            mlflow.autolog(
                log_input_examples=True,
                log_model_signatures=True,
                log_models=True,
                log_datasets=True,
            )

            training_parameters = {
                "n_estimators": 100,  # the number of rounds to run the training
                "max_depth": 3,  # maximum depth of a tree
                "eta": 0.5,  # step size shrinkage used in updates to prevent overfitting
                "alpha": 2.5,  # L1 regularization term on weights
                "objective": "reg:squarederror",
                "eval_metric": "rmse",  # evaluation metrics for validation data
                "subsample": 0.8,  # subsample ratio of the training instance
                "colsample_bytree": 0.8,  # subsample ratio of columns when constructing each tree
                "min_child_weight": 3,  # minimum sum of instance weight (hessian) needed in a child
                "early_stopping_rounds": 10,  # the model trains until it stops improving.
                "verbosity": 1,  # verbosity of printing messages
            }
            model = xgb.XGBRegressor(**training_parameters)

            with tempfile.TemporaryDirectory() as tmp_dir:
                mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path="encoders", dst_path=tmp_dir
                )
                preprocessor = joblib.load(
                    f"{tmp_dir}/encoders/categorical_encoder.joblib"
                )

            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", model)]
            )

            train_df = pd.read_csv(train_data)
            y_train = np.array(train_df.loc[:, target])
            train_df.drop([target], axis=1, inplace=True)
            print("train_df.columns:", train_df.columns)

            validation_df = pd.read_csv(validation_data)
            y_validation = np.array(validation_df.loc[:, target])
            validation_df = preprocessor.transform(validation_df)
            validation_df.drop([f"remainder__{target}"], axis=1, inplace=True)
            print("validation_df.columns:", validation_df.columns)

            pipeline.fit(
                train_df, y_train, model__eval_set=[(validation_df, y_validation)]
            )

            return {
                "experiment_name": experiment_name,
                "run_id": run_id,
                "training_run_id": training_run_id,
            }
