import pandas as pd

import predict as p


def history(n=500):
    return pd.read_csv(p.OUT / "clean_energy.csv", index_col=0, parse_dates=True).tail(n)


def test_predict_shape():
    fc = p.predict(history())
    assert len(fc) == p.HORIZON
    assert fc.notna().all()
    assert fc.index.is_monotonic_increasing


def test_short_history_raises():
    try:
        p.predict(history(100))
    except ValueError:
        return
    raise AssertionError("expected ValueError for short history")


def test_monthly_bill():
    tiers = p.load_config()["tiers"]
    assert p.monthly_bill(0, tiers) == 0
    assert abs(p.monthly_bill(50, tiers) - 34.0) < 1e-9
    assert abs(p.monthly_bill(100, tiers) - 77.5) < 1e-9


if __name__ == "__main__":
    test_predict_shape()
    print("predict shape: ok")
    test_short_history_raises()
    print("short history: ok")
    test_monthly_bill()
    print("monthly bill: ok")
