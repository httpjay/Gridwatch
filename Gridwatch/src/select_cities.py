import pandas as pd

df = pd.read_csv("../data/processed/base_city_dataset.csv")

sample_df = df.head(50).copy()

print("Selected cities shape:", sample_df.shape)
print(sample_df.head())

sample_df.to_csv("../data/processed/sample_cities_50.csv", index=False)

print("Saved sample cities to ../data/processed/sample_cities_50.csv")