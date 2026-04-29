import pandas as pd

# Load raw weather forecast data
df = pd.read_csv("../data/raw/weather_90011_nws_forecast.csv")

print("Original shape:", df.shape)
print(df.head())

# Keep only useful columns
df = df[["name", "startTime", "endTime", "temperature", "temperatureUnit", "shortForecast"]]

# Rename columns
df = df.rename(columns={
    "name": "period_name",
    "temperature": "temperature",
    "temperatureUnit": "temperature_unit",
    "shortForecast": "condition"
})

# Add location column
df["location"] = "90011"

# Reorder columns
df = df[["location", "period_name", "startTime", "endTime", "temperature", "temperature_unit", "condition"]]

print("\nCleaned shape:", df.shape)
print(df.head())

# Save cleaned weather file
df.to_csv("../data/processed/weather_90011_cleaned.csv", index=False)

print("Saved cleaned weather data to ../data/processed/weather_90011_cleaned.csv")