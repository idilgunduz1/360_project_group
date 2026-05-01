import pandas as pd
import numpy as np
from datetime import datetime
from forecast_helper import get_price_data, get_historical_weather, get_weather_forecast
import statsmodels.api as sm

###
### MODEL: Weather Linear Regression (OLS)
###
### Features: temperature at Istanbul & Ankara, hour of day, lagged price (48 h).
### Train OLS on the full history; predict hours 0-23 of tomorrow.
### Uses statsmodels OLS — same library as the provided forecast.py reference.
### Note: electricity prices CAN be negative — do not clip predictions.
###

# ---------------------------------------------------------------------------
# 1. DATE SETUP
# ---------------------------------------------------------------------------

date_to_be_forecasted = str(datetime.now() + pd.DateOffset(1))[:10]
# date_to_be_forecasted = "2026-05-05"  # uncomment to back-test a past day

last_datetime_to_be_forecasted = (
    pd.to_datetime(date_to_be_forecasted) + pd.DateOffset(days=1, hours=-1)
)

print("Current date              :", datetime.now())
print("Last hour to be forecasted:", last_datetime_to_be_forecasted)


# ---------------------------------------------------------------------------
# 2. LOAD PRICE DATA
# ---------------------------------------------------------------------------

price_data = get_price_data()


# ---------------------------------------------------------------------------
# 3. FETCH WEATHER DATA
# ---------------------------------------------------------------------------

start_date = "2024-01-01"

plant_coordinates = [
    [41.0082, 28.9784],   # Istanbul (major demand center)
    [39.9334, 32.8597],   # Ankara (major demand center)
]
weather_variables = ["temperature_2m"]
weather_forecast_models = ["ecmwf_ifs025"]

meteo_data_historical = get_historical_weather(
    start_date=start_date,
    variables=weather_variables,
    coordinates=plant_coordinates,
    get_forecast_data=True,
)

meteo_data_future = get_weather_forecast(
    forecast_days=5,
    past_days=30,
    variables=weather_variables,
    coordinates=plant_coordinates,
    models=weather_forecast_models,
)

meteo_data_historical = meteo_data_historical.dropna()
meteo_data_future     = meteo_data_future.dropna()

### Combine: prefer historical data for overlapping dates, use forecast for future.
meteo_data_historical.insert(0, "type", "historical")
meteo_data_future.insert(0, "type", "future")
meteo_data_all = pd.concat([meteo_data_historical, meteo_data_future], axis=0)
meteo_data_all["priorty"] = meteo_data_all.groupby("dt")["type"].rank(ascending=False)
meteo_data_all = meteo_data_all[meteo_data_all["priorty"] == 1]
meteo_data_all = meteo_data_all.sort_values("dt")
meteo_data_all = meteo_data_all.drop(["type", "priorty"], axis=1)


# ---------------------------------------------------------------------------
# 4. BUILD A CONTINUOUS HOURLY GRID
# ---------------------------------------------------------------------------

### Gap-free grid ensures .shift(N) is always exactly N hours.

df_dates = pd.date_range(start=start_date, end=last_datetime_to_be_forecasted, freq="1h")
df_dates = df_dates.tz_localize("Europe/Istanbul")

df = pd.DataFrame({"dt": df_dates})


# ---------------------------------------------------------------------------
# 5. JOIN PRICE AND WEATHER DATA
# ---------------------------------------------------------------------------

df = df.merge(price_data[["dt", "price"]], on="dt", how="left")
df = df.merge(meteo_data_all, on="dt", how="left")


# ---------------------------------------------------------------------------
# 6. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

### price_lag_48: prices are published with a ~2-day delay by EPIAS.
### ffill propagates the last known value into tomorrow's rows so the
### prediction slice always has a valid feature.
df["price_lag_48"] = df["price"].shift(48).ffill()   # 48 = 2 days * 24 hours

df["hour"] = df["dt"].dt.hour  # 0-23, captures intraday price patterns

### Weather column names from the helper: "location_000 temperature_2m", etc.
weather_cols = [c for c in df.columns if "temperature_2m" in c]


# ---------------------------------------------------------------------------
# 7. TRAIN OLS MODEL
# ---------------------------------------------------------------------------

feature_cols = ["price_lag_48", "hour"] + weather_cols

### Drop rows where price OR any feature is NaN — these become training data.
### The last 24 rows (tomorrow) have NaN price and stay in df as prediction rows.
train_X = df.dropna()[feature_cols]
train_y = df.dropna()["price"]

print(f"Training rows: {len(train_X)} | Features: {train_X.columns.tolist()}")

train_X_const = sm.add_constant(train_X)
model   = sm.OLS(train_y, train_X_const)
results = model.fit()


# ---------------------------------------------------------------------------
# 8. PREDICT NEXT DAY
# ---------------------------------------------------------------------------

next_day_X = df.iloc[-24:][feature_cols].copy()
next_day_X_const = sm.add_constant(next_day_X)

predictions = results.predict(next_day_X_const).tolist()

print("Predictions:", predictions)


# ---------------------------------------------------------------------------
# 9. OUTPUT — must be the very last line
# ---------------------------------------------------------------------------

print(predictions)
