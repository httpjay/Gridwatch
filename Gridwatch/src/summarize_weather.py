import pandas as pd

# Load cleaned weather data
df = pd.read_csv("../data/processed/weather_90011_cleaned.csv")

print("Input shape:", df.shape)
print(df.head())

# Make sure temperature is numeric
df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")

# Create summary features
summary = pd.DataFrame([{
    "zcta": "90011",
    "avg_temp_forecast": df["temperature"].mean(),
    "max_temp_forecast": df["temperature"].max(),
    "min_temp_forecast": df["temperature"].min()
}])

print("\nWeather summary:")
print(summary)

# Save summary file
summary.to_csv("../data/processed/weather_90011_summary.csv", index=False)

print("Saved weather summary to ../data/processed/weather_90011_summary.csv")