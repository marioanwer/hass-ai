# HASS AI Add-ons

**HASS (Home Automation Smart System)** is an ESP32 smart-home project with a Flutter app and a Firebase Realtime Database. This repository holds the AI add-ons built on top of it.

| Feature | Task | Status |
|---|---|---|
| Energy data cleaning and exploration | 1 | done |
| Consumption and bill forecasting | 2 | done |
| Appliance fault detection and usage-habit learning | 3 | done |
| Voice control (English, then Arabic) | 4, 5 | not covered in this README yet |
| Firebase integration, logger, slides | 6 | not covered in this README yet |

**Contents:** [Layout](#repository-layout) · [Setup](#setup) · [Datasets](#datasets) · [Task 1](#task-1-clean-and-explore-the-energy-dataset) · [Task 2](#task-2-consumption-and-bill-forecasting) · [Task 3](#task-3-fault-detection-and-usage-habit-learning) · [What is simulated](#honest-reporting-what-is-simulated-and-what-is-public)

---

## Repository layout

```
hass-ai/
  energy/                         Tasks 1-3: notebooks, scripts, models
    01_clean_explore.ipynb        Task 1 notebook (writes clean_energy.csv)
    02_forecast.ipynb             Task 2 notebook
    03_anomaly_habits.ipynb       Task 3 notebook
    clean_energy.csv              hourly table from Task 1 (input of Tasks 2 and 3)
    data_dictionary.md            columns and cleaning decisions (written by Task 1)
    predict.py                    next-24-hour forecast (function + command line)
    anomaly_habits.py             detect(df) and suggest_rules(df)
    forecast_model.joblib         Task 2 model
    anomaly_model.joblib          Task 3 fault detector (Isolation Forest + rules)
    habit_kmeans.joblib           Task 3 KMeans models (not used for the suggestions, see Task 3)
    config.json                   electricity tariff tiers
    *.png                         figures used in this README
  Augmented Smart Home with Weather Information/   raw data (Mendeley)
  individual+household+electric+power+consumption/ raw data (UCI)
  speech_commands/, arabic-speech-commands-dataset-v1.0/   voice datasets (Tasks 4-5)
  environment.yml
  README.md
```

Data is not committed (see `.gitignore`). Small result files such as `clean_energy.csv` can be allowed explicitly if the team wants them in the repository.

## Setup

```bash
mamba env create -f environment.yml
conda activate hass-ai
```

Manual alternative (energy tasks):

```bash
mamba create -n hass-ai python=3.11 -y
conda activate hass-ai
mamba install -c conda-forge numpy pandas matplotlib seaborn scikit-learn scipy statsmodels xgboost lightgbm joblib jupyterlab ipykernel -y
```

Download the datasets (links below) into the folders shown in the layout. The notebooks find the repository root from the current folder, so no paths need editing. Open them from the repository root or from `energy/` and use **Restart Kernel and Run All**.

**Run order:** `01_clean_explore.ipynb` first (it creates `energy/clean_energy.csv`), then `02_forecast.ipynb` and `03_anomaly_habits.ipynb` in any order.

## Datasets

| Dataset | Used by | Link | License / credit |
|---|---|---|---|
| Individual Household Electric Power Consumption (UCI) | Tasks 1, 2, 3 | https://archive.ics.uci.edu/dataset/235 | Hebrail and Berard |
| Augmented Smart Home with Weather Information (Mendeley Data) | Task 3 (habits) | https://data.mendeley.com/datasets/pxnb7gh646/1 | CC0 |
| Smart Home Dataset with Weather Information (Kaggle, taranvee), optional | Task 1 second look | https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information | see Kaggle page |
| Google Speech Commands v0.02 | Tasks 4, 5 | https://huggingface.co/datasets/google/speech_commands | CC BY 4.0 |
| Arabic Speech Commands | Tasks 4, 5 | https://zenodo.org/records/4662481 | CC BY 4.0 |

Each energy dataset is a different house, so they are never merged: every model is trained on one dataset.

---

## Task 1: Clean and explore the energy dataset

**Goal.** One clean, documented hourly file of household consumption that Tasks 2 and 3 use, and an understanding of how consumption behaves over the day, week and season.

**Dataset.** UCI Individual Household Electric Power Consumption: one house in France, one reading per minute, December 2006 to November 2010, 2,075,259 rows and 9 columns.

**Deliverables.** `energy/clean_energy.csv`, `energy/01_clean_explore.ipynb` (profiles, monthly trend, correlation heatmap) and `energy/data_dictionary.md`. The notebook rebuilds the same file from the raw data and checks that the result is byte-identical.

### Cleaning decisions
1. The file is semicolon-separated and uses `?` for missing values.
2. `Date` and `Time` are combined into one datetime index (day first, `dd/mm/yyyy`).
3. 25,979 minute rows (1.25%) are missing, with the same count in each of the seven measurement columns.
4. Data is resampled from one minute to one hour (mean): 34,589 hours in the range, 421 of them without any reading.
5. Gaps of 3 hours or less are forward-filled (this affected 1 hour); longer gaps are dropped (420 hours). Result: **34,169 hours** from 2006-12-16 17:00 to 2010-11-26 21:00, no missing values. The dropped hours are gaps in the time index.
6. Added features: `unmetered`, `hour`, `weekday` (0 = Monday), `month`, `is_weekend`.

### Data dictionary

| Column | Unit | Meaning |
|---|---|---|
| `datetime` (index) | | start of the hour |
| `Global_active_power` | kW | household active power, hourly mean of the minute values |
| `Global_reactive_power` | kW | household reactive power, hourly mean |
| `Voltage` | V | hourly mean voltage |
| `Global_intensity` | A | hourly mean current intensity |
| `Sub_metering_1` | Wh per minute | kitchen (dishwasher, oven, microwave), hourly mean |
| `Sub_metering_2` | Wh per minute | laundry room (washing machine, dryer, refrigerator, lights), hourly mean |
| `Sub_metering_3` | Wh per minute | electric water heater and air conditioner, hourly mean |
| `unmetered` | Wh per minute | `Global_active_power * 1000 / 60` minus the three sub-meters: consumption they do not cover |
| `hour` | 0-23 | hour of day |
| `weekday` | 0-6 | day of week, 0 = Monday |
| `month` | 1-12 | calendar month |
| `is_weekend` | 0 or 1 | 1 on Saturday and Sunday |

### Findings
Write the 5 to 8 sentences from the notebook's last analysis cell here (peak hour, weekday vs weekend, winter vs summer, share of the sub-meters).

---

## Task 2: Consumption and bill forecasting

**Goal.** Forecast household consumption for the next 24 hours from past consumption and calendar features, and turn the forecast into an estimated monthly bill.

### Data
`energy/clean_energy.csv` from Task 1 (UCI, hourly). Target: `Global_active_power` (kW, hourly mean, so numerically kWh per hour).

### Pipeline (`energy/02_forecast.ipynb`)
1. Load the file and put the hours removed in Task 1 back as missing (`asfreq("1h")`), so that "24 hours ago" really means the same hour yesterday. Rows that touch a gap are dropped when the features are built.
2. Features: hour, weekday, is_weekend and past values of the target only (lags and rolling means). The other columns of the file (voltage, current, sub-meters) are measured in the same hour that is being predicted, so using them would leak the answer.
3. Time-based split, never random. Train: before 2009-12-01. Validation: 2009-12-01 to 2010-05-31. Test: from 2010-06-01 to the end of the data (2010-11-26).
4. Baselines first (same hour yesterday, same hour last week, previous hour). Random Forest and XGBoost with a small grid tuned on validation only. Final models retrained on train + validation; the test set was evaluated once.
5. Two models: a **next-hour** model (uses lags of 1, 2, 24, 168 hours) and a **day-ahead** model (only lags of 24 hours or more, because the last 23 hours are unknown when forecasting a full day). The day-ahead Random Forest is the one saved and used by `predict.py`.

### Results (test set)

**Day-ahead model (published, used by `predict.py`)**

| Model | MAE (kW) | RMSE (kW) |
|---|---|---|
| Baseline: same hour yesterday | 0.4870 | 0.7335 |
| Baseline: same hour last week | 0.5425 | 0.7886 |
| **Random Forest, day-ahead** | **0.4234** | **0.5767** |

It beats the yesterday baseline by about 13% (MAE) and 21% (RMSE). Most important features: `lag_168` (36%), `hour` (19%), `lag_24` (14%), `roll_168` (10%), `roll_24` (8%), `lag_48` (8%).

**Next-hour model (secondary result)**

| Model | MAE (kW) | RMSE (kW) |
|---|---|---|
| Baseline: previous hour | 0.3810 | 0.5769 |
| Baseline: same hour yesterday | 0.4879 | 0.7345 |
| Random Forest | 0.3169 | 0.4608 |
| XGBoost | 0.3144 | 0.4595 |

These improve on the previous-hour baseline by about 17% (MAE). They rely mostly on `lag_1` (71% of importance), so compare them with the previous-hour baseline, not with the day-ahead results, which predict a longer horizon.

The validation period (December to May) has higher and less regular consumption than the test period, so its errors are higher (day-ahead Random Forest MAE 0.4947 vs 0.6550 for the yesterday baseline).

![Actual vs predicted, first test week](energy/actual_vs_predicted_week.png)

### Monthly bill estimate
Over the first 30 days of the test period (720 hourly rows) the day-ahead model predicted **728.4 kWh** against **698.1 kWh** actual (4.3% error). With the tiered tariff in `energy/config.json` this is **1,275.20 EGP predicted vs 1,204.10 EGP actual** (5.9% error). Each tier is priced only for the kWh inside its range, and fixed service fees are not included. Hourly errors partly cancel when summed, so this is a monthly estimate, not hourly accuracy. The bill error is larger than the kWh error because the higher tiers amplify it.

The data is from a French house while the tariff is an Egyptian tier table, so the amount is an illustration of the method. Tariff prices and rules differ between sources and change over time: edit `config.json` to match your own bill.

### Run
```bash
cd energy
python predict.py path/to/history.csv
```
The CSV needs the time as the first column, a `Global_active_power` column (kW), hourly rows and at least 192 hours. It prints the next 24 hours of forecast. The notebook checks that the output matches the notebook model on a test day.

### Limitations
- One home, about 4 years of data, recorded in France.
- The model was tuned on December to May and tested on June to November, so it mainly shows how well the method generalizes to other seasons within the same house.
- No weather features: the UCI data has none, and HASS would need weather forecasts to use them.
- The tariff is configurable but approximate.
- A model trained on the UCI house will not fit the HASS home as is. Retrain on the data logged from the real hardware (Task 6).

---

## Task 3: Fault detection and usage-habit learning

Notebook: `energy/03_anomaly_habits.ipynb`. Functions: `energy/anomaly_habits.py`.

### A. Habit learning

**Data.** Augmented Smart Home with Weather Information (Mendeley), hourly (8,399 rows). Here the per-device `[kW]` columns are used, because the UCI data only has three sub-meters and no per-device columns.

**Findings.**
- `Air conditioning [kW]` correlates 0.99 with temperature and varies only 2% over the day, so it is a function of temperature rather than a usage habit. `Outdoor lights [kW]` is flat (0.4% variation). Both are excluded. The other devices have a correlation with temperature between -0.01 and 0.04.
- KMeans (k from 2 to 5) on each device's 24-hour daily profile (349 days) gives silhouette scores of 0.05 to 0.16 for the best k, so days do not form distinct groups, and the groups are not explained by weekends or temperature. The suggestions therefore come from each device's **average daily profile**, not from the clusters. The KMeans models are saved in `habit_kmeans.joblib` only because the task asks for them.
- Weekday and weekend profiles are nearly identical (maximum gap 0.02 to 0.05 of the device range, split threshold 0.2), so every suggestion applies to every day.

**Rule.** A device "peaks" during the hours where its average load is above the midpoint between its lowest and highest hourly average. No device drops to zero in this data, so "peak" is relative.

![Daily profiles, weekday vs weekend](energy/device_daily_profiles.png)

**Suggestions produced by `suggest_rules(df)`:**
- Water heater peaks from 06:00 to 09:00 and from 20:00 to 23:00 every day.
- Pool Pump peaks from 09:00 to 14:00 and from 17:00 to 22:00 every day.
- Laundry peaks from 10:00 to 12:00 and from 20:00 to 22:00 every day.
- Home Theater peaks from 18:00 to 01:00 (next day) every day.
- Car charger peaks from 21:00 to 10:00 (next day) every day.

**Caveat.** The `[kW]` device columns were added to the dataset and look synthetic: the car charger averages 2.0 kW, more than the whole-house mean of 0.86 kW, and the daily curves repeat almost identically every day. Treat the habits as a demonstration of the method, not as measurements of a real home.

### B. Fault detection

**Data.** `energy/clean_energy.csv` from Task 1 (UCI, 34,169 hours, 2006-12-16 to 2010-11-26).

**Split and simulated faults.** The first 70% is used for training and the last 30% for testing. **No real faults exist in the data**, so 40 faults were injected into a copy of the test set, each on consecutive hours:
- 20 spikes: one hour with 4 kW added to the total consumption.
- 10 stuck-on faults: `Sub_metering_3` held at its 99th percentile for 12 hours.
- 10 outages: total consumption and all sub-meters set to zero from 18:00 to 21:00.

An event counts as detected when at least one of its hours is flagged.

**Model.** Isolation Forest (300 trees, contamination 0.01) plus three simple rules. Features: single-hour power and `Sub_metering_3`, the ratio and difference to the average of the same hour in the previous 7 days, the change versus the previous 3 hours, and the 6-hour minimum and standard deviation of `Sub_metering_3`. An hour is flagged if the Isolation Forest or any rule fires. The rule thresholds come from the normal training data only (0.1% and 99.9% quantiles):

| Rule | Fires when | Targets |
|---|---|---|
| `low_power` | consumption falls far below the usual level for that hour | outages |
| `power_jump` | consumption is far above the usual level for that hour | spikes |
| `stuck_high` | `Sub_metering_3` stays above its limit for 6 hours | stuck-on devices |

**Why the rules.** The Isolation Forest alone found 24 of 40 events: it missed every outage (0/10), because the training data never drops below 0.124 kW and tree models do not extrapolate, so a zero is scored as normal; and it found only half of the stuck-on faults.

Isolation Forest alone, by contamination:

| Contamination | Precision | Recall (hours) | False-alarm rate | Spikes | Stuck-on | Outages |
|---|---|---|---|---|---|---|
| 0.005 | 0.360 | 0.100 | 0.003 | 17/20 | 1/10 | 0/10 |
| **0.010** | 0.259 | 0.167 | 0.009 | 19/20 | 5/10 | 0/10 |
| 0.020 | 0.183 | 0.222 | 0.018 | 19/20 | 5/10 | 0/10 |
| 0.050 | 0.143 | 0.483 | 0.052 | 20/20 | 10/10 | 0/10 |

Published detector (contamination 0.01 + rules):

| Method | Precision | Recall (hours) | False-alarm rate | Spikes | Stuck-on | Outages |
|---|---|---|---|---|---|---|
| Isolation Forest only | 0.259 | 0.167 | 0.009 | 19/20 | 5/10 | 0/10 |
| Rule `low_power` only | 1.000 | 0.222 | 0.000 | 0/20 | 0/10 | 10/10 |
| Rule `power_jump` only | 0.933 | 0.078 | 0.0001 | 14/20 | 0/10 | 0/10 |
| Rule `stuck_high` only | 0.222 | 0.400 | 0.025 | 0/20 | 10/10 | 0/10 |
| **Isolation Forest + rules** | **0.295** | **0.728** | **0.031** | **19/20** | **10/10** | **10/10** |

**39 of 40 faults are detected.** The false-alarm rate is 3.1% of normal hours (roughly 0.75 false alarm hours per day), most of it from the `stuck_high` rule. Precision is low because faults are only 1.76% of the hours: even a small false-alarm rate produces more false alarms than real ones. The hour-level recall (0.73) is lower than the event-level result because the stuck-on rule needs 6 hours to fire and the first hours of each fault are missed.

![Injected faults vs alerts](energy/anomaly_timeline.png)

**Run.**
```bash
cd energy
python anomaly_habits.py detect path/to/hourly.csv
python anomaly_habits.py habits path/to/habits_hourly.csv
```
From Python: `detect(df)` returns one row per hour with `anomaly` (Isolation Forest or any rule), `score`, `iso`, `has_context` and one column per rule; `suggest_rules(df)` returns the suggestions as text. The notebook checks that `detect()` gives exactly the same 444 flagged hours as the notebook, and that `suggest_rules()` gives the same text. On the real UCI file (no injected faults), the flags are not confirmed faults.

### Limitations
- Faults are simulated and their types and sizes were chosen by us. The results show that faults of these shapes can be detected, not that real faults will be.
- The spike size was changed from "multiply by 4" to "add 4 kW" after the first run, because a multiplied low reading (for example at night) is still a normal value and is not a fault. The rules and features were chosen after looking at results on the same test set (no separate validation set), so the numbers are optimistic.
- The false-alarm rate of 3.1% is above the 2% target we wanted. It can be reduced by raising the `stuck_high` limit, which should be tuned on a validation part of the training data.
- The detector is trained on UCI columns (active power and sub-meters) that the HASS hardware does not produce. It must be retrained on the data logged from the real device (Task 6).
- The habit data looks synthetic (see part A).

---

## Honest reporting: what is simulated and what is public

- **Simulated:** all faults used to evaluate the fault detector (Task 3).
- **Possibly synthetic:** the per-device `[kW]` columns of the Mendeley dataset (Task 3, part A).
- **Public data:** everything else (UCI, Mendeley, Kaggle, Google Speech Commands, Arabic Speech Commands).
- Data logged from the real HASS hardware (Firebase) is planned for Task 6 and not used yet.

## Next steps
- Task 6: log real HASS data and retrain the Task 2 and Task 3 models on it.
- Tune the `stuck_high` rule on a validation split to lower the false-alarm rate.
- Add the sections for Tasks 4 to 6.