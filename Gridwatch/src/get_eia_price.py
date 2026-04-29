import requests
import pandas as pd

API_KEY = "ELzyUCEHUttMnO7uuLB2rcha1JUDXzWHrekaZ19H"

url = "https://api.eia.gov/v2/electricity/retail-sales/data/"

params = {
    "api_key": API_KEY,
    "frequency": "monthly",
    "data[0]": "price",
    "facets[stateid][]": "CA",
    "facets[sectorid][]": "RES",
    "start": "2023-01",
    "end": "2024-12",
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "length": 5000
}

response = requests.get(url, params=params)

print("Status:", response.status_code)

data = response.json()

records = data["response"]["data"]

df = pd.DataFrame(records)

print(df.head())

df = df[["period", "stateDescription", "price"]]
df["price"] = pd.to_numeric(df["price"], errors="coerce")

df.to_csv("../data/raw/eia_ca_price.csv", index=False)

print("Saved EIA price data to ../data/raw/eia_ca_price.csv")