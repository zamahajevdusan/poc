import pandas as pd
import joblib
import numpy as np # type ignore
import xgboost as xgb # type ignore
from sklearn.pipeline import Pipeline
import valohai

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

preprocessor = joblib.load(
    valohai.inputs("preprocessor").path()
)

# preprocessor = joblib.load(
#     "categorical_encoder.joblib"
# )

pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", model)]
)

train_data = valohai.inputs("train_data").path()
validation_data = valohai.inputs("validation_data").path()

# train_data = "train.csv"
# validation_data = "validation_data.csv"

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

joblib.save(pipeline, "/valohai/outputs/pipeline.joblib")

