"""
demo_data.py
============
Creates a SYNTHETIC (computer-generated) weather dataset for testing the application.

*** This is NOT real weather data. ***
It follows a made-up seasonal pattern (hot spring, monsoon-style summer rain) plus random
noise, so every chart has something to show. A few flaws are injected ON PURPOSE so the
cleaning step has something to detect:

    * 2 exact duplicate rows
    * 4 missing humidity values and 3 missing temperature values
    * 1 non-numeric rainfall entry ("error")
    * 1 impossible humidity (140 %) and 1 impossible temperature (75 C)
    * 1 extreme but possible temperature (58 C, flagged as an outlier and kept)
    * 1 row whose minimum and maximum temperature are swapped

Run `python demo_data.py` to write data/demo_weather_SYNTHETIC.csv.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

DEMO_FILE_NAME = "demo_weather_SYNTHETIC.csv"

_RAIN_CHANCE = {1: 0.03, 2: 0.03, 3: 0.04, 4: 0.07, 5: 0.12, 6: 0.45,
                7: 0.60, 8: 0.60, 9: 0.50, 10: 0.25, 11: 0.07, 12: 0.03}


def generate_demo_dataframe(seed: int = 42, start: str = "2023-01-01",
                            end: str = "2024-12-31") -> pd.DataFrame:
    """Return the synthetic dataset as it would look when read from a CSV file."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    day_of_year = dates.dayofyear.to_numpy()
    month = dates.month.to_numpy()

    chance = np.array([_RAIN_CHANCE[m] for m in month])
    scale = np.where(np.isin(month, [6, 7, 8, 9]), 14.0, 6.0)
    rain = np.where(rng.random(n) < chance, rng.exponential(scale), 0.0).round(1)

    temperature = (27.5 + 6.5 * np.sin(2 * np.pi * (day_of_year - 35) / 365.25)
                   + rng.normal(0, 1.5, n) - 1.5 * (rain > 0))
    humidity = (62 - 1.8 * (temperature - 27.5) + 18 * (rain > 0) + rng.normal(0, 6, n))
    humidity = np.clip(humidity, 15, 100).round(1)
    wind = np.clip(8 + 3 * np.sin(2 * np.pi * (day_of_year - 150) / 365.25)
                   + rng.gamma(2.0, 1.2, n), 0, None).round(1)
    t_min = (temperature - 4 - np.abs(rng.normal(0, 1, n))).round(1)
    t_max = (temperature + 4 + np.abs(rng.normal(0, 1, n))).round(1)

    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Temperature (C)": temperature.round(1),
        "Min Temperature (C)": t_min,
        "Max Temperature (C)": t_max,
        "Rainfall (mm)": rain,
        "Humidity (%)": humidity,
        "Wind Speed (km/h)": wind,
    })
    df["Rainfall (mm)"] = df["Rainfall (mm)"].astype(object)

    # ---- deliberate flaws (fixed positions so results are reproducible) ----
    df.loc[[40, 140, 240, 340], "Humidity (%)"] = np.nan
    df.loc[[60, 160, 460], "Temperature (C)"] = np.nan
    df.loc[75, "Rainfall (mm)"] = "error"
    df.loc[90, "Humidity (%)"] = 140.0
    df.loc[120, "Temperature (C)"] = 75.0
    df.loc[150, "Temperature (C)"] = 58.0
    low, high = df.loc[200, "Min Temperature (C)"], df.loc[200, "Max Temperature (C)"]
    df.loc[200, "Min Temperature (C)"] = high
    df.loc[200, "Max Temperature (C)"] = low
    duplicates = df.iloc[[10, 300]]
    df = pd.concat([df.iloc[:11], duplicates.iloc[[0]], df.iloc[11:301],
                    duplicates.iloc[[1]], df.iloc[301:]], ignore_index=True)
    return df


def demo_csv_bytes() -> bytes:
    """The demo dataset as CSV bytes (same format an uploaded file would have)."""
    return generate_demo_dataframe().to_csv(index=False).encode("utf-8")


if __name__ == "__main__":
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, DEMO_FILE_NAME)
    with open(path, "wb") as handle:
        handle.write(demo_csv_bytes())
    print(f"Wrote {path}")
