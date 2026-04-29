import pandas as pd

df = pd.read_csv("../data/processed/city_master_dataset_50.csv")
print("Shape:", df.shape)
print(df.head())