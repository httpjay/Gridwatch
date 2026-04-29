import pandas as pd
import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the 50-city dataset
df = pd.read_csv("../data/processed/city_master_dataset_50.csv")

features = [
    "household_income",
    "avg_annual_energy_cost",
    "avg_temp_forecast",
    "max_temp_forecast",
    "min_temp_forecast"
]

X = df[features]
y = df["risk_label"]

print("Dataset shape:", df.shape)
print("Risk label counts:")
print(y.value_counts())

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

print("Type of shap_values:", type(shap_values))

if isinstance(shap_values, list):
    print("SHAP returned a list")
    for i, sv in enumerate(shap_values):
        print(f"Class {i} shape:", np.array(sv).shape)
    values_to_use = np.array(shap_values[1] if len(shap_values) > 1 else shap_values[0])

else:
    shap_values = np.array(shap_values)
    print("SHAP array shape:", shap_values.shape)

    if shap_values.ndim == 2:
        values_to_use = shap_values
    elif shap_values.ndim == 3:
        # shape is (rows, features, classes) -> choose class 1
        values_to_use = shap_values[:, :, 1]
    else:
        raise ValueError(f"Unexpected SHAP shape: {shap_values.shape}")

print("Using SHAP values shape:", values_to_use.shape)

# Build dataframe
shap_df = pd.DataFrame(values_to_use, columns=features)
shap_df["city"] = df["city"]

print("\nSHAP dataframe preview:")
print(shap_df.head())

# Save file
output_path = "../data/processed/shap_values_50.csv"
shap_df.to_csv(output_path, index=False)

print(f"\nSaved SHAP values to {output_path}")