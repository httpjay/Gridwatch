import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("../data/processed/city_master_dataset_50.csv")

print("Dataset shape:", df.shape)

# Features
X = df[[
    "household_income",
    "avg_annual_energy_cost",
    "avg_temp_forecast",
    "max_temp_forecast",
    "min_temp_forecast"
]]

# Target
y = df["risk_label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))