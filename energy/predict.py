import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).with_name("forecast_model.joblib")
MIN_HISTORY = 192


def predict(history, col="Global_active_power"):
    bundle = joblib.load(MODEL_PATH)
    model, feats = bundle["model"], bundle["features"]

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

    h = pd.Timedelta(hours=1)
    idx = pd.date_range(y.index[-1] + h, periods=24, freq="1h")

    rows = []
    for t in idx:
        rows.append({
            "hour": t.hour,
            "weekday": t.day_of_week,
            "is_weekend": int(t.day_of_week >= 5),
            "lag_24": y.loc[t - 24 * h],
            "lag_48": y.loc[t - 48 * h],
            "lag_168": y.loc[t - 168 * h],
            "roll_24": y.loc[t - 47 * h: t - 24 * h].mean(),
            "roll_168": y.loc[t - 191 * h: t - 24 * h].mean(),
        })
    X = pd.DataFrame(rows, index=idx)[feats]
    return pd.Series(model.predict(X), index=idx, name="forecast_kW")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python predict.py history.csv")
    hist = pd.read_csv(sys.argv[1], index_col=0, parse_dates=True)
    print(predict(hist).round(3).to_string())
