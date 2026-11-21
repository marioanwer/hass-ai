import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from predict import CONFIG_PATH, FEATURES, OUT, TARGET, make_features

CLEAN_CSV = OUT / "clean_energy.csv"
TEST_START = pd.Timestamp("2010-06-01")
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=10, max_depth=15,
                 n_jobs=-1, random_state=42)
DEFAULT_CFG = {
    "currency": "EGP",
    "tiers": [
        {"up_to_kwh": 50, "price": 0.68},
        {"up_to_kwh": 100, "price": 0.87},
        {"up_to_kwh": 200, "price": 1.00},
        {"up_to_kwh": 350, "price": 1.73},
        {"up_to_kwh": 650, "price": 2.18},
        {"up_to_kwh": 1000, "price": 2.35},
        {"up_to_kwh": None, "price": 2.89},
    ],
}


def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"{name:34s} MAE={mae:.4f}  RMSE={rmse:.4f}")


def main():
    df = pd.read_csv(CLEAN_CSV, index_col=0, parse_dates=True).asfreq("1h")
    base = make_features(df[TARGET])
    base[TARGET] = df[TARGET]
    base = base.dropna()

    trva = base[base.index < TEST_START]
    te = base[base.index >= TEST_START]
    print(f"train+val: {len(trva)} rows | test: {len(te)} rows")

    model = RandomForestRegressor(**RF_PARAMS).fit(trva[FEATURES], trva[TARGET])

    report("RF day-ahead (test)", te[TARGET], model.predict(te[FEATURES]))
    report("Baseline yesterday (test)", te[TARGET], te["lag_24"])
    report("Baseline last week (test)", te[TARGET], te["lag_168"])

    joblib.dump({"model": model, "features": FEATURES,
                 "sklearn_version": sklearn.__version__},
                OUT / "forecast_model.joblib")
    print("saved:", OUT / "forecast_model.joblib")

    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CFG, indent=2), encoding="utf-8")
        print("created default:", CONFIG_PATH)


if __name__ == "__main__":
    main()
