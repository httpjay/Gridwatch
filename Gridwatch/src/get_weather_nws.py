import requests
import pandas as pd

# Approximate point for ZIP 90011 / South Los Angeles
lat = 34.0072
lon = -118.2587

headers = {
    "User-Agent": "GridWatch project (jpancha1@lion.lmu.edu)",
    "Accept": "application/geo+json"
}

# Step 1: Get forecast grid info for the point
points_url = f"https://api.weather.gov/points/{lat},{lon}"
points_response = requests.get(points_url, headers=headers, timeout=30)

print("Points status:", points_response.status_code)
points_data = points_response.json()

forecast_url = points_data["properties"]["forecast"]
print("Forecast URL:", forecast_url)

# Step 2: Get forecast data
forecast_response = requests.get(forecast_url, headers=headers, timeout=30)
print("Forecast status:", forecast_response.status_code)

forecast_data = forecast_response.json()
periods = forecast_data["properties"]["periods"]

df = pd.DataFrame(periods)

# Keep useful columns
df = df[[
    "number",
    "name",
    "startTime",
    "endTime",
    "temperature",
    "temperatureUnit",
    "windSpeed",
    "windDirection",
    "shortForecast",
    "detailedForecast"
]]

print(df.head())

df.to_csv("../data/raw/weather_90011_nws_forecast.csv", index=False)
print("Saved to ../data/raw/weather_90011_nws_forecast.csv")