import pandas as pd

df = pd.read_csv("../data/raw/lead_energy_data.csv", skiprows=8)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nHead:")
print(df.head())