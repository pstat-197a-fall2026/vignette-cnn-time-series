from pathlib import Path
import pandas as pd

project_data = Path(__file__).resolve().parents[1] / "data"
data_dir = project_data / "USArray_Ground_Temperature_1680" / "data"
file = data_dir / "H17K-1_2017-07-11_2019-06-29.csv"

raw = pd.read_csv(file)

print(raw.shape)
print(raw.head(3).to_string(index=False))
# Using a depth of .2
soil = raw[["date_time", "tsoil_0.2"]].copy()
soil = soil.rename(columns={"tsoil_0.2": "temp_c"})

print(soil.head(3))
print(soil.dtypes)

soil["date_time"] = pd.to_datetime(soil["date_time"], format="%Y-%m-%d %H:%M")
soil = soil.sort_values("date_time")

print(soil["date_time"].dtype)
print(soil["date_time"].diff().value_counts())

daily = (
    soil.resample("D", on="date_time")["temp_c"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "temp_c", "count": "n_readings"})
)

print(daily.head())
print(daily.loc[daily["n_readings"] != 4])

daily_clean = daily.loc[daily["n_readings"] == 4, ["temp_c"]].copy()
print(daily_clean.shape)

output_file = project_data / "H17K-1_0.2m_daily.csv"
daily_clean.to_csv(output_file, index_label="date")
print(f"Saved {len(daily_clean)} daily averages to {output_file}")
