# Data dictionary: clean_energy.csv

Hourly table with 34,169 rows from 2006-12-16 17:00:00 to 2010-11-26 21:00:00. The first column is the datetime index (`datetime`). Built by the notebook `01_clean_explore.ipynb` from the UCI Individual Household Electric Power Consumption file.

| Column | Unit | Meaning |
|---|---|---|
| `Global_active_power` | kW | household active power, hourly mean of the minute values |
| `Global_reactive_power` | kW | household reactive power, hourly mean |
| `Voltage` | V | hourly mean voltage |
| `Global_intensity` | A | hourly mean current intensity |
| `Sub_metering_1` | Wh per minute | kitchen (dishwasher, oven, microwave), hourly mean |
| `Sub_metering_2` | Wh per minute | laundry room (washing machine, dryer, refrigerator, lights), hourly mean |
| `Sub_metering_3` | Wh per minute | electric water heater and air conditioner, hourly mean |
| `unmetered` | Wh per minute | Global_active_power * 1000 / 60 minus the three sub-meters: consumption they do not cover |
| `hour` | 0-23 | hour of day |
| `weekday` | 0-6 | day of week, 0 = Monday |
| `month` | 1-12 | calendar month |
| `is_weekend` | 0 or 1 | 1 on Saturday and Sunday |

## Cleaning decisions

1. The raw file is semicolon-separated and `?` means missing: 25,979 of 2,075,259 minute rows have no reading.
2. Date and Time are combined into one datetime index (day first).
3. Minutes are averaged into hours: 34,589 hours in range, 421 of them without any reading. The first and last hours are averages of fewer than 60 minutes.
4. Gaps of up to 3 hours are forward-filled (1 hours). Longer gaps are dropped (420 hours), so the time index has gaps there.
5. Result: 34,169 hours and no missing values.
6. Added columns: `unmetered`, `hour`, `weekday`, `month`, `is_weekend`.
