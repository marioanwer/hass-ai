"""clean_energy.py: clean the UCI Individual Household Electric Power Consumption data.

Reads the semicolon-separated minute file ('?' marks a missing value), builds an
hourly table without missing values and adds calendar features.

    python clean_energy.py household_power_consumption.txt clean_energy.csv

Cleaning decisions:
  * '?' is read as missing.
  * Date and Time are combined into one datetime index (day first).
  * Minutes are averaged into hours. An hour without any reading stays missing.
  * Missing runs of up to MAX_FILL_HOURS hours are forward-filled; longer runs are dropped.

The same functions can be reused on other data with the same columns, for example
the readings logged from the real HASS hardware.
"""
import sys

import pandas as pd

MAX_FILL_HOURS = 3
TARGET = "Global_active_power"
SUBS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]


def read_raw(path):
    """Minute-level data indexed by datetime, every column as float."""
    raw = pd.read_csv(path, sep=";", na_values=["?"], low_memory=False)
    raw.index = pd.to_datetime(raw["Date"] + " " + raw["Time"], format="%d/%m/%Y %H:%M:%S")
    raw.index.name = "datetime"
    return raw.drop(columns=["Date", "Time"]).astype(float).sort_index()


def to_hourly(raw, max_fill_hours=MAX_FILL_HOURS):
    """Hourly means; short gaps are forward-filled, long gaps are dropped.

    Returns (hourly table, dict with the counts behind every decision).
    """
    hourly = raw.resample("1h").mean()

    isna = hourly[TARGET].isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    short = isna & (run_len <= max_fill_hours)
    hourly = hourly.where(~short, hourly.ffill())

    stats = {
        "raw_rows": len(raw),
        "raw_missing_rows": int(raw[TARGET].isna().sum()),
        "hours_in_range": len(hourly),
        "hours_without_data": int(isna.sum()),
        "hours_filled": int(short.sum()),
    }
    n_before = len(hourly)
    hourly = hourly.dropna()
    stats["hours_dropped"] = n_before - len(hourly)
    stats["hours_final"] = len(hourly)
    return hourly, stats


def add_features(hourly):
    """Calendar features and the energy not covered by the three sub-meters."""
    out = hourly.copy()
    out["unmetered"] = out[TARGET] * 1000 / 60 - out[SUBS].sum(axis=1)   # Wh per minute
    out["hour"] = out.index.hour
    out["weekday"] = out.index.day_of_week                              # 0 = Monday
    out["month"] = out.index.month
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)
    return out


def clean(path, max_fill_hours=MAX_FILL_HOURS):
    """Raw file -> (clean hourly table, stats)."""
    hourly, stats = to_hourly(read_raw(path), max_fill_hours)
    return add_features(hourly), stats


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python clean_energy.py household_power_consumption.txt clean_energy.csv")
    table, info = clean(sys.argv[1])
    table.to_csv(sys.argv[2], lineterminator="\n")
    print(info)
    print("saved:", sys.argv[2], table.shape)
