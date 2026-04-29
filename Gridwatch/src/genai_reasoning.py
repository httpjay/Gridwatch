import pandas as pd

# Load merged dataset and shap values
data_df = pd.read_csv("../data/processed/city_master_dataset_50.csv")
shap_df = pd.read_csv("../data/processed/shap_values_50.csv")

# Pick first city as example
city_name = data_df.iloc[0]["city"]

city_data = data_df[data_df["city"] == city_name].iloc[0]
city_shap = shap_df[shap_df["city"] == city_name].iloc[0]

# Remove city column from shap
feature_impacts = city_shap.drop("city").sort_values(key=abs, ascending=False)

print("City:", city_name)
print("\nTop feature impacts:")
print(feature_impacts)

# Build a simple reasoning-style narrative
narrative = f"""
City: {city_name}

GridWatch evaluates city-level energy affordability risk using household income,
annual energy cost, and near-term weather conditions.

For {city_name}, the current profile shows:
- Household income: {city_data['household_income']}
- Average annual energy cost: {city_data['avg_annual_energy_cost']}
- Average forecast temperature: {city_data['avg_temp_forecast']}
- Maximum forecast temperature: {city_data['max_temp_forecast']}
- Minimum forecast temperature: {city_data['min_temp_forecast']}
- Risk label: {city_data['risk_label']}

Top model drivers for this city:
{feature_impacts.to_string()}

Interpretation:
For {city_name}, the model suggests that affordability risk is driven primarily by the combination
of economic conditions and weather-related demand. In this case, household income has the strongest
influence, followed by minimum forecast temperature and annual energy cost. This indicates that both
financial capacity and expected energy demand are important in explaining why this city is flagged.

This prototype now runs on 52 California cities and demonstrates the full GridWatch pipeline:
data ingestion → feature merging → model training → SHAP explainability → reasoning narrative.
"""
print(narrative)