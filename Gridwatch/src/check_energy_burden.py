import pandas as pd

df = pd.read_csv("../data/processed/city_master_dataset_50.csv")

print("Rows:", len(df))
print("\nEnergy burden summary:")
print(df["energy_burden"].describe())

print("\nTop 20 highest energy burden cities:")
print(df[["city", "energy_burden"]].sort_values("energy_burden", ascending=False).head(20))

print("\nCount above thresholds:")
for t in [4, 5, 6, 7, 8, 9, 10]:
    count = (df["energy_burden"] > t).sum()
    print(f"> {t}: {count}")