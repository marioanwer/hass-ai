import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

OUT = Path(__file__).resolve().parent
MODEL_PATH = OUT / "forecast_model.joblib"
CONFIG_PATH = OUT / "config.json"
TARGET = "Global_active_power"
MIN_HISTORY = 192
HORIZON = 24
FEATURES = ["hour", "weekday", "is_weekend", "lag_24", "lag_48", "lag_168", "roll_24", "roll_168"]


def make_features(y):
    f = pd.DataFrame(index=y.index)
    f["hour"] = y.index.hour
    f["weekday"] = y.index.day_of_week
    f["is_weekend"] = (f["weekday"] >= 5).astype(int)
    for lag in [24, 48, 168]:
        f[f"lag_{lag}"] = y.shift(lag)
    f["roll_24"] = y.shift(24).rolling(24).mean()
    f["roll_168"] = y.shift(24).rolling(168).mean()
    return f


def predict(history, col=TARGET):
    bundle = joblib.load(MODEL_PATH)

    saved = bundle.get("sklearn_version")
    if saved:
        import sklearn
        if saved != sklearn.__version__:
            warnings.warn(f"model saved with scikit-learn {saved}, "
                          f"running {sklearn.__version__}")

    y = history[col] if isinstance(history, pd.DataFrame) else history
    y = y.astype(float).sort_index().asfreq("1h").ffill()
    if len(y) < MIN_HISTORY:
        raise ValueError(f"need at least {MIN_HISTORY} hours of history, got {len(y)}")

    future = pd.date_range(y.index[-1] + pd.Timedelta(hours=1), periods=HORIZON, freq="1h")
    ext = y.reindex(y.index.append(future))
    X = make_features(ext).loc[future, bundle["features"]]
    return pd.Series(bundle["model"].predict(X), index=future, name="forecast_kW")


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def monthly_bill(kwh, tiers):
    total, lower = 0.0, 0.0
    for t in tiers:
        upper = float("inf") if t["up_to_kwh"] is None else t["up_to_kwh"]
        total += max(0.0, min(kwh, upper) - lower) * t["price"]
        lower = upper
        if kwh <= upper:
            break
    return total


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict.py history.csv")
    hist = pd.read_csv(sys.argv[1], index_col=0, parse_dates=True)
    fc = predict(hist)
    print(fc.round(3).to_string())
    print(f"next 24 hours: {fc.sum():.2f} kWh")


if __name__ == "__main__":
    main()
