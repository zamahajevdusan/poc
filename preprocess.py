import joblib # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.preprocessing import OrdinalEncoder # type: ignore
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
target_col = "actual_shipping_days"

# Encode categorical features
cat_pipeline = ColumnTransformer(
    [
        (
            "cat",
            OrdinalEncoder(handle_unknown="error"),
            categorical_features_names,
        )
    ],
    remainder="passthrough",
).set_output(transform="pandas")

input_file = valohai.inputs("input_file").path()

df_model_data = pd.read_csv(input_file)

encoder = cat_pipeline.fit(df_model_data)
joblib.dump(encoder, "/valohai/outputs/categorical_encoder.joblib")

# Shuffle and splitting dataset
train_data, validation_data, test_data = np.split(
    df_model_data.sample(frac=1, random_state=1729),
    [int(0.7 * len(df_model_data)), int(0.9 * len(df_model_data))],
)

print("train_data:\n", train_data)
print("validation_data:\n", validation_data)
print("test_data:\n", test_data)

train_data.to_csv("/valohai/outputs/train.csv", index=False)
validation_data.to_csv("/valohai/outputs/validation_data.csv", index=False)
test_data.to_csv("/valohai/outputs/test_data.csv", index=False)

