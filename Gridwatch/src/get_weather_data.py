import requests
import pandas as pd
import os


TOKEN = "xmfGrbpjDTkcuzRqWvDCVJcYwGxotWvB"

url = "https://www.ncei.noaa.gov/cdo-web/api/v2/stations"
headers = {"token": TOKEN}

params = {
    "datasetid": "GHCND",
    "datatypeid": "TMAX",
    "extent": "33.95,-118.30,34.08,-118.15",   # smaller LA box
    "limit": 5,
    "sortfield": "name",
    "sortorder": "asc"
}

try:
    response = requests.get(url, headers=headers, params=params, timeout=(10, 60))
    print("Status:", response.status_code)
    data = response.json()
    results = data.get("results", [])

    print("Stations found:", len(results))
    for s in results:
        print(s["id"], "|", s["name"])

except requests.exceptions.RequestException as e:
    print("Request failed:", e)