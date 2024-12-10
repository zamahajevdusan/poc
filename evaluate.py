import pandas as pd
import joblib
import xdg
import valohai

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


test_df = pd.read_csv(valohai.inputs("test_data").path())
model = joblib.load(valohai.inputs("model").path())

print("test_df.shape:", test_df.shape)
print("test_df.columns:", test_df.columns)




