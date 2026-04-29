import requests
import pandas as pd
import time

coords_df = pd.read_csv("../data/processed/city_coordinates_50.csv")

headers = {
    "User-Agent": "GridWatch project (jpancha1@lion.lmu.edu)",
    "Accept": "application/geo+json"
}

weather_rows = []

for _, row in coords_df.iterrows():
    city = row["city"]
    state = row["state"]
    lat = row["latitude"]
    lon = row["longitude"]

    try:
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        points_response = requests.get(points_url, headers=headers, timeout=30)

        if points_response.status_code != 200:
            print(f"Failed points request for {city}: {points_response.status_code}")
            continue

        forecast_url = points_response.json()["properties"]["forecast"]

        forecast_response = requests.get(forecast_url, headers=headers, timeout=30)

        if forecast_response.status_code != 200:
            print(f"Failed forecast request for {city}: {forecast_response.status_code}")
            continue

        periods = forecast_response.json()["properties"]["periods"]
        forecast_df = pd.DataFrame(periods)

        forecast_df["temperature"] = pd.to_numeric(forecast_df["temperature"], errors="coerce")

        weather_rows.append({
            "city": city,
            "state": state,
            "avg_temp_forecast": forecast_df["temperature"].mean(),
            "max_temp_forecast": forecast_df["temperature"].max(),
            "min_temp_forecast": forecast_df["temperature"].min()
        })

        print(f"Success: {city}")
        time.sleep(1)

    except Exception as e:
        print(f"Error for {city}: {e}")

weather_summary_df = pd.DataFrame(weather_rows)

print("\nWeather summary shape:", weather_summary_df.shape)
print(weather_summary_df.head())

weather_summary_df.to_csv("../data/processed/city_weather_summary_50.csv", index=False)

print("Saved city weather summary to ../data/processed/city_weather_summary_50.csv")