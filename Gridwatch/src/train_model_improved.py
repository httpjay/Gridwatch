import os
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("../data/processed/city_master_dataset_50.csv")

print("Loaded dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

# -----------------------------
# Required columns
# -----------------------------
feature_cols = [
    "energy_burden",
    "household_income",
    "avg_annual_energy_cost",
    "avg_temp_forecast",
    "max_temp_forecast",
    "min_temp_forecast"
]

target_col = "risk_label"

missing_cols = [col for col in feature_cols + [target_col] if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# -----------------------------
# Keep only needed columns
# -----------------------------
data = df[feature_cols + [target_col]].dropna().copy()

print("Shape after dropna:", data.shape)
print("Risk label distribution:")
print(data[target_col].value_counts())

if len(data) < 10:
    raise ValueError(f"Dataset too small after preprocessing: {len(data)} rows")

# -----------------------------
# Convert target to int
# -----------------------------
data[target_col] = data[target_col].astype(int)

X = data[feature_cols]
y = data[target_col]

# Stratify only if both classes exist
stratify_arg = y if y.nunique() > 1 else None

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_arg
)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# -----------------------------
# Model + hyperparameter tuning
# -----------------------------
base_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

param_grid = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.1],
    "n_estimators": [50, 100],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# -----------------------------
# Evaluation
# -----------------------------
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nBest Params:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save model
# -----------------------------
os.makedirs("../models", exist_ok=True)
model_path = "../models/xgboost_gridwatch.json"
best_model.save_model(model_path)

print(f"\nModel saved to: {model_path}")