from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from anomaly_habits import HABIT_DEVICES, apply_rules, build_context_features

OUT = Path(__file__).resolve().parent
REPO = OUT.parent
CLEAN_CSV = OUT / "clean_energy.csv"
HABITS_FILE = (REPO / "Augmented Smart Home with Weather Information"
               / "Augmented Smart Home with Weather Information" / "HomeC_augmented.csv")

BIG_DEVICES = ["Air conditioning [kW]", "Water heater [kW]", "Car charger [kW]",
               "Pool Pump [kW]", "Outdoor lights [kW]", "Home Theater [kW]", "Laundry [kW]"]
SUBS = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
BLOCK = 48
PLAN = ["spike"] * 20 + ["stuck_on"] * 10 + ["drop_to_zero"] * 10
RULE_Q = 0.001
CONTAMINATION = 0.01


def load_hourly_clean(path=CLEAN_CSV):
    hourly = pd.read_csv(path, index_col=0, parse_dates=True)
    needed = ["Global_active_power"] + SUBS + ["unmetered", "hour", "is_weekend"]
    missing = [c for c in needed if c not in hourly.columns]
    assert not missing, f"columns missing from clean_energy.csv: {missing}"
    assert hourly.isna().sum().sum() == 0
    assert hourly.index.is_monotonic_increasing and not hourly.index.has_duplicates
    return hourly


def load_habits(path=HABITS_FILE):
    df = pd.read_csv(path, low_memory=False,
                     usecols=["time", "use_HO", "temperature"] + BIG_DEVICES)
    t_text = pd.to_datetime(df["time"], errors="coerce")
    if t_text.notna().mean() > 0.9:
        df["time"] = t_text
    else:
        df["time"] = pd.to_datetime(pd.to_numeric(df["time"], errors="coerce"), unit="s")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    assert len(df) > 0, "no valid time values found in the habits data"
    df = df.apply(pd.to_numeric, errors="coerce").resample("1h").mean()
    df["hour"] = df.index.hour
    df["weekday"] = df.index.day_of_week
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df


def inject_faults(train, test_clean, seed=42):
    rng = np.random.default_rng(seed)
    test = test_clean.copy()
    j = {c: test.columns.get_loc(c) for c in test.columns}
    labels = np.zeros(len(test), dtype=int)
    kinds = np.array([""] * len(test), dtype=object)
    events = []

    blocks = rng.choice(len(test) // BLOCK, size=len(PLAN), replace=False)
    stuck_level = train["Sub_metering_3"].quantile(0.99)
    hours = test["hour"].to_numpy()

    for kind, b in zip(PLAN, blocks):
        base = int(b) * BLOCK
        if kind == "spike":
            start = end = base + int(rng.integers(0, 40))
            test.iloc[start, j["Global_active_power"]] += 4.0
        elif kind == "stuck_on":
            start = base + int(rng.integers(0, 30))
            end = start + 11
            sl = slice(start, end + 1)
            s3 = test.iloc[sl, j["Sub_metering_3"]].to_numpy()
            delta = np.clip(stuck_level - s3, 0, None)
            test.iloc[sl, j["Sub_metering_3"]] = s3 + delta
            test.iloc[sl, j["Global_active_power"]] = (
                test.iloc[sl, j["Global_active_power"]].to_numpy() + delta * 0.06)
        else:
            start = base + int(np.argmax(hours[base:base + BLOCK] == 18))
            end = start + 3
            assert hours[start] == 18, "no 18:00 inside the block"
            for c in ["Global_active_power"] + SUBS:
                test.iloc[start:end + 1, j[c]] = 0.0
        labels[start:end + 1] = 1
        kinds[start:end + 1] = kind
        events.append({"kind": kind, "start": start, "end": end})

    test["unmetered"] = test["Global_active_power"] * 1000 / 60 - test[SUBS].sum(axis=1)
    return test, labels, kinds, events, stuck_level


def evaluate(pred, labels, events):
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    row = {
        "precision": tp / max(tp + fp, 1),
        "recall_hours": tp / max(tp + fn, 1),
        "false_alarm_rate": fp / int((labels == 0).sum()),
    }
    for kind in ["spike", "stuck_on", "drop_to_zero"]:
        ev = [e for e in events if e["kind"] == kind]
        caught = sum(bool(pred[e["start"]:e["end"] + 1].any()) for e in ev)
        row[f"{kind}_caught"] = f"{caught}/{len(ev)}"
    return row


def daily_matrix(df, col):
    m = df[col].groupby([df.index.date, df.index.hour]).mean().unstack().dropna()
    m.index = pd.to_datetime(m.index)
    return m


def fit_habit_models(habits):
    models = {}
    for dev in HABIT_DEVICES:
        M = daily_matrix(habits, dev)
        mean, std = float(M.values.mean()), float(M.values.std())
        X = (M - mean) / std
        km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
        models[dev] = {"model": km, "scale": (mean, std),
                       "silhouette": float(silhouette_score(X, km.labels_))}
    return models


def main():
    hourly = load_hourly_clean()
    cut = int(len(hourly) * 0.70)
    train = hourly.iloc[:cut].copy()
    test_clean = hourly.iloc[cut:].copy()
    print(f"train: {len(train)} rows  {train.index.min()} -> {train.index.max()}")
    print(f"test : {len(test_clean)} rows  {test_clean.index.min()} -> {test_clean.index.max()}")

    test, labels, kinds, events, stuck_level = inject_faults(train, test_clean)
    print("injected hours:", int(labels.sum()), f"({labels.mean() * 100:.2f}% of test)")
    print("stuck_on level (Wh/min):", round(float(stuck_level), 2))

    full = pd.concat([train, test])
    F = build_context_features(full)
    X_train = F.loc[train.index].dropna()
    X_test = F.loc[test.index]
    n_filled = int(X_test.isna().any(axis=1).sum())
    X_test = X_test.fillna(X_train.median())
    print(f"train rows: {len(X_train)} | test rows with missing context (filled): {n_filled}")

    rule_thr = {
        "GAP_ratio_low": X_train["GAP_ratio"].quantile(RULE_Q),
        "GAP_diff_high": X_train["GAP_diff"].quantile(1 - RULE_Q),
        "SM3_min6_high": X_train["SM3_min6"].quantile(1 - RULE_Q),
    }
    print("rule thresholds:", {k: round(float(v), 3) for k, v in rule_thr.items()})

    iso = IsolationForest(n_estimators=300, contamination=CONTAMINATION,
                          random_state=42, n_jobs=-1).fit(X_train)
    iso_flag = iso.predict(X_test) == -1
    rules = apply_rules(X_test, rule_thr)
    pred = (iso_flag | rules.any(axis=1).to_numpy()).astype(int)

    print("\nIsolation Forest alone:")
    print(pd.Series(evaluate(iso_flag.astype(int), labels, events)).to_string())
    print("\nIsolation Forest + rules:")
    print(pd.Series(evaluate(pred, labels, events)).to_string())
    print("\nflagged test hours:", int(pred.sum()))

    bundle = {
        "model": iso,
        "features": list(F.columns),
        "rule_thresholds": {k: float(v) for k, v in rule_thr.items()},
        "fill_values": X_train.median().to_dict(),
        "contamination": CONTAMINATION,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, OUT / "anomaly_model.joblib")
    print("saved:", OUT / "anomaly_model.joblib")

    habits = load_habits()
    habit_models = fit_habit_models(habits)
    for dev, info in habit_models.items():
        print(f"{dev:20s} silhouette={info['silhouette']:.3f}")
    joblib.dump(habit_models, OUT / "habit_kmeans.joblib")
    print("saved:", OUT / "habit_kmeans.joblib")


if __name__ == "__main__":
    main()
