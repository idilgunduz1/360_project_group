import pandas as pd
import numpy as np
from datetime import datetime
from forecast_helper import get_price_data

###
### MODEL: Differenced Lag (Trend-Adjusted Seasonal Naive)
###
### predicted_price = price_lag_168 + (price_lag_168 - price_lag_336)
###   lag_168 = same hour 1 week ago   (168 = 7 * 24)
###   lag_336 = same hour 2 weeks ago  (336 = 14 * 24)
###
### The difference (lag_168 - lag_336) captures the week-over-week trend.
### If lag_336 is unavailable (e.g. at the start of the dataset), fall back
### to plain seasonal naive (just lag_168).
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

### Gap-free grid so .shift(N) always means exactly N hours.

start_date = "2024-01-01"

df_dates = pd.date_range(start=start_date, end=last_datetime_to_be_forecasted, freq="1h")
df_dates = df_dates.tz_localize("Europe/Istanbul")

df = pd.DataFrame({"dt": df_dates})


# ---------------------------------------------------------------------------
# 4. JOIN PRICE DATA
# ---------------------------------------------------------------------------

df = df.merge(price_data[["dt", "price"]], on="dt", how="left")


# ---------------------------------------------------------------------------
# 5. LAG FEATURES
# ---------------------------------------------------------------------------

df["price_lag_168"] = df["price"].shift(168)   # 168 = 7 days * 24 hours
df["price_lag_336"] = df["price"].shift(336)   # 336 = 14 days * 24 hours


# ---------------------------------------------------------------------------
# 6. DIFFERENCED-LAG FORECAST
# ---------------------------------------------------------------------------

### Week-over-week change: if prices rose 50 TRY last week vs. two weeks ago,
### we expect them to rise another 50 TRY this week.
trend = df["price_lag_168"] - df["price_lag_336"]

df["predicted_price"] = np.where(
    df["price_lag_336"].notna(),
    df["price_lag_168"] + trend,  # trend-adjusted when 2-week lag is available
    df["price_lag_168"],          # plain lag-168 fallback
)


# ---------------------------------------------------------------------------
# 7. EXTRACT NEXT-DAY PREDICTIONS
# ---------------------------------------------------------------------------

next_day = df.iloc[-24:]

### Electricity prices CAN be negative — do not clip.
fallback = float(df["price"].median())
predictions = next_day["predicted_price"].fillna(fallback).tolist()

print("Predictions:", predictions)


# ---------------------------------------------------------------------------
# 8. OUTPUT — must be the very last line
# ---------------------------------------------------------------------------

print(predictions)
