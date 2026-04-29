import pandas as pd

# Load datasets
census_df = pd.read_csv("../data/processed/census_cleaned.csv", dtype={"zcta": str})
weather_df = pd.read_csv("../data/processed/weather_90011_summary.csv", dtype={"zcta": str})
eia_df = pd.read_csv("../data/processed/eia_price_summary.csv", dtype={"zcta": str})

# Keep only 90011 from census
census_90011 = census_df[census_df["zcta"] == "90011"]

print("Census row:")
print(census_90011)

print("\nWeather row:")
print(weather_df)

print("\nEIA row:")
print(eia_df)

# Merge census + weather
merged_df = pd.merge(census_90011, weather_df, on="zcta", how="inner")

# Merge result + eia
merged_df = pd.merge(merged_df, eia_df, on="zcta", how="inner")

print("\nFinal merged dataset:")
print(merged_df)

# Save final merged dataset
merged_df.to_csv("../data/processed/master_dataset_v2.csv", index=False)

print("\nSaved final merged dataset to ../data/processed/master_dataset_v2.csv")