import pandas as pd

df = pd.read_csv("../data/raw/eia_ca_price.csv")

summary = pd.DataFrame([{
    "zcta": "90011",
    "avg_electricity_price": df["price"].mean(),
    "max_electricity_price": df["price"].max()
}])

print(summary)

summary.to_csv("../data/processed/eia_price_summary.csv", index=False)

print("Saved EIA summary")