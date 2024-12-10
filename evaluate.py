import pandas as pd
import mlflow

target = "actual_shipping_days"
categorical_features_names = [
    "carrier",
    "in_bulk_order",
    "shipping_origin",
    "on_time_delivery",
    "computer_brand",
    "shipping_priority",
    "computer_model",
]


def run(test_data, tracking_server_arn, experiment_name, run_id, training_run_id):

    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="ModelEvaluation", nested=True):
            test_df = pd.read_csv(test_data)

            print("test_df.shape:", test_df.shape)
            print("test_df.columns:", test_df.columns)

            ml_pipeline = mlflow.pyfunc.load_model(f"runs:/{training_run_id}/model")

            results = mlflow.evaluate(
                model=ml_pipeline,
                data=test_df,
                targets=target,
                model_type="regressor",
                evaluators=["default"],
            )
            return {"mean_squared_error": results.metrics["mean_squared_error"]}
