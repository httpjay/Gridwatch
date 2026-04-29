import pandas as pd

df = pd.read_csv("../data/processed/city_master_dataset_v1.csv")

# example value (use your computed average)
avg_price = 0.28  # update if you calculated differently

df["avg_electricity_price"] = avg_price

print(df.head())

df.to_csv("../data/processed/city_master_dataset_v2.csv", index=False)

print("Saved dataset with electricity price")