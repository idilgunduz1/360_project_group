import pandas as pd
import numpy as np
from datetime import datetime
from forecast_helper import get_price_data

###
### MODEL: Seasonal Naive (Same-Hour-Last-Week)
###
### For each of the 24 hours we predict, look up the actual price from the
### same clock-hour exactly one week ago (168 hours).
### "Seasonal" here means the weekly cycle of electricity prices.
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
# 3. BUILD A CONTINUOUS HOURLY GRID
# ---------------------------------------------------------------------------

### We build a gap-free hourly index so that .shift(N) always means N hours.
### Without this, any missing hour in the raw data would misalign the lags.

start_date = "2024-01-01"

df_dates = pd.date_range(start=start_date, end=last_datetime_to_be_forecasted, freq="1h")
df_dates = df_dates.tz_localize("Europe/Istanbul")

df = pd.DataFrame({"dt": df_dates})


# ---------------------------------------------------------------------------
# 4. JOIN PRICE DATA
# ---------------------------------------------------------------------------

df = df.merge(price_data[["dt", "price"]], on="dt", how="left")


# ---------------------------------------------------------------------------
# 5. LAG FEATURE
# ---------------------------------------------------------------------------

df["price_lag_168"] = df["price"].shift(168)   # 168 = 7 days * 24 hours


# ---------------------------------------------------------------------------
# 6. EXTRACT NEXT-DAY PREDICTIONS
# ---------------------------------------------------------------------------

next_day = df.iloc[-24:]  # last 24 rows = hours 0-23 of the day to be forecasted

### If the lag value is missing (e.g. data gap), fall back to the median price.
fallback = float(df["price"].median())
predictions = next_day["price_lag_168"].fillna(fallback).tolist()

print("Predictions:", predictions)


# ---------------------------------------------------------------------------
# 7. OUTPUT — must be the very last line
# ---------------------------------------------------------------------------

print(predictions)
