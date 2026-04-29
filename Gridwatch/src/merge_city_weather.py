import pandas as pd

base_df = pd.read_csv("../data/processed/base_city_dataset.csv")
weather_df = pd.read_csv("../data/processed/city_weather_summary_50.csv")
sample_df = pd.read_csv("../data/processed/sample_cities_50.csv")

print("Base shape:", base_df.shape)
print("Weather shape:", weather_df.shape)
print("Sample shape:", sample_df.shape)

base_selected = pd.merge(
    sample_df[["city", "state"]],
    base_df,
    on=["city", "state"],
    how="left"
)

print("Base selected shape:", base_selected.shape)

merged_df = pd.merge(
    base_selected,
    weather_df,
    on=["city", "state"],
    how="inner"
)

print("Merged shape:", merged_df.shape)
print(merged_df.head())

merged_df.to_csv("../data/processed/city_master_dataset_50.csv", index=False)

print("Saved merged city dataset to ../data/processed/city_master_dataset_50.csv")