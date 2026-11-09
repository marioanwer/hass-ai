from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).with_name("anomaly_model.joblib")

HABIT_DEVICES = ["Water heater [kW]", "Pool Pump [kW]", "Laundry [kW]",
                 "Home Theater [kW]", "Car charger [kW]"]


# ---------------- fault detection ----------------
def _expected_same_hour(s, days=7, min_days=4):
    lags = pd.concat([s.shift(24 * k) for k in range(1, days + 1)], axis=1)
    return lags.mean(axis=1).where(lags.notna().sum(axis=1) >= min_days)


def apply_rules(X, thr):
    return pd.DataFrame({
        "low_power":  X["GAP_ratio"] < thr["GAP_ratio_low"],    # consumption collapses vs usual
        "power_jump": X["GAP_diff"] > thr["GAP_diff_high"],     # far above the usual for this hour
        "stuck_high": X["SM3_min6"] > thr["SM3_min6_high"],     # device never drops for 6 hours
    }, index=X.index)


def detect(df, model_path=MODEL_PATH):
    """Isolation Forest + three simple rules. Returns one row per hour with data."""
    bundle = joblib.load(model_path)
    if "rule_thresholds" not in bundle:
        raise ValueError("anomaly_model.joblib has no rule_thresholds: re-run cell 13 of the notebook")
    d = df.sort_index()
    grid = pd.date_range(d.index.min(), d.index.max(), freq="1h")
    d = d.reindex(grid)
    gap = d["Global_active_power"].astype(float)
    sm3 = d["Sub_metering_3"].astype(float)
    exp_gap = _expected_same_hour(gap)

    F = pd.DataFrame(index=grid)
    F["GAP"] = gap
    F["GAP_ratio"] = gap / (exp_gap + 0.1)
    F["GAP_diff"] = gap - exp_gap
    F["GAP_ratio_prev3"] = gap / (gap.shift(1).rolling(3).mean() + 0.1)
    F["SM3"] = sm3
    F["SM3_min6"] = sm3.rolling(6).min()
    F["SM3_std6"] = sm3.rolling(6).std()
    F["hour"] = grid.hour
    F = F[bundle["features"]]

    valid = gap.notna() & sm3.notna()
    has_context = F.notna().all(axis=1)
    X = F.fillna(pd.Series(bundle["fill_values"]))[valid]

    iso_flag = bundle["model"].predict(X) == -1
    rules = apply_rules(X, bundle["rule_thresholds"])
    out = pd.DataFrame({
        "anomaly": iso_flag | rules.any(axis=1).to_numpy(),
        "score": bundle["model"].score_samples(X),
        "iso": iso_flag,
        "has_context": has_context[valid],
    }, index=X.index)
    return out.join(rules)


# ---------------- habit suggestions ----------------
def high_windows(profile, frac=0.5):
    p = profile.reindex(range(24))
    on = (p >= p.min() + frac * (p.max() - p.min())).to_numpy()
    if on.all() or not on.any():
        return []
    wins = []
    for h in range(24):
        if on[h] and not on[h - 1]:
            end = h
            while on[(end + 1) % 24]:
                end += 1
            wins.append((h, (end + 1) % 24))
    return wins


def _fmt(h):
    return f"{h:02d}:00"


def _by_hour(s):
    return s.groupby(s.index.hour).mean().reindex(range(24))


def suggest_rules(df, devices=HABIT_DEVICES, frac=0.5, split_tol=0.2, verbose=False):
    if "is_weekend" not in df.columns:
        df = df.assign(is_weekend=(df.index.day_of_week >= 5).astype(int))
    out = []
    for dev in devices:
        name = dev.replace(" [kW]", "")
        p_all = _by_hour(df[dev])
        p_wd = _by_hour(df.loc[df["is_weekend"] == 0, dev])
        p_we = _by_hour(df.loc[df["is_weekend"] == 1, dev])

        gap = (p_wd - p_we).abs().max() / (p_all.max() - p_all.min())
        if verbose:
            print(f"{name:18s} weekday-vs-weekend gap = {gap:.2f}")

        if gap < split_tol:
            groups = [("every day", p_all)]
        else:
            groups = [("on weekdays", p_wd), ("on weekends", p_we)]

        for when, prof in groups:
            for s, e in high_windows(prof, frac):
                end = _fmt(e) + (" (next day)" if e <= s else "")
                out.append(f"{name} peaks from {_fmt(s)} to {end} {when}. "
                           f"Create an automatic schedule?")
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3 or sys.argv[1] not in ("detect", "habits"):
        sys.exit("Usage: python anomaly_habits.py detect|habits hourly.csv")
    d = pd.read_csv(sys.argv[2], index_col=0, parse_dates=True)
    if sys.argv[1] == "detect":
        r = detect(d)
        print("flagged hours:", int(r["anomaly"].sum()), "of", len(r))
        print(r[r["anomaly"]].tail(10).round(3).to_string())
    else:
        print("\n".join(suggest_rules(d)))
