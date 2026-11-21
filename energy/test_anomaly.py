import anomaly_habits
from train_anomaly import load_habits, load_hourly_clean

REQUIRED_COLUMNS = {"anomaly", "score", "iso", "has_context",
                    "low_power", "power_jump", "stuck_high"}


def test_detect_output():
    hourly = load_hourly_clean()
    result = anomaly_habits.detect(hourly)
    assert len(result) > 0
    assert REQUIRED_COLUMNS <= set(result.columns)
    assert result["anomaly"].dtype == bool
    assert result["score"].notna().all()


def test_suggest_rules():
    lines = anomaly_habits.suggest_rules(load_habits())
    assert len(lines) > 0
    assert all(line.endswith("Create an automatic schedule?") for line in lines)


if __name__ == "__main__":
    test_detect_output()
    print("detect: ok")
    test_suggest_rules()
    print("suggest_rules: ok")
