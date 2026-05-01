import pandas as pd
import numpy as np
from datetime import datetime
from forecast_helper import get_price_data,get_historical_weather,get_weather_forecast
import statsmodels.api as sm


# Adjust the forecasting day.
### date_to_be_forecasted defines the day that your model should predict.
### Testing whether your model works well for the past days might be useful.

date_to_be_forecasted = str(datetime.now()+pd.DateOffset(1))[:10]
# date_to_be_forecasted = "2026-05-05"
last_datetime_to_be_forecasted = pd.to_datetime(date_to_be_forecasted)+pd.DateOffset(days=1,hours=-1)

current_date = datetime.now().date()
print("Current date:",datetime.now())
print("Last date and hour to be forecasted:",last_datetime_to_be_forecasted)

# Get price data (dependent variable)
price_data = get_price_data()


# Get weather data
### Here we set get_forecast_data True. It means that we use forecast history as past data.
### You can set it False if you want to use actual historical weather data.
### Weather can influence electricity demand which in turn affects electricity prices.
start_date = "2024-01-01"
plant_coordinates = [
                        [41.0082, 28.9784],   # Istanbul (major demand center)
                        [39.9334, 32.8597],   # Ankara (major demand center)
                    ]
weather_forecast_models=["ecmwf_ifs025"]
weather_variables=["temperature_2m"]


meteo_data_historical = get_historical_weather(start_date=start_date,
                                    variables=weather_variables,
                                    coordinates=plant_coordinates,
                                    get_forecast_data=True,)

meteo_data_future = get_weather_forecast(forecast_days=5,
                                         past_days=30,
                                         variables=weather_variables,
                                         coordinates=plant_coordinates,
                                         models=weather_forecast_models,)


meteo_data_historical = meteo_data_historical.dropna()
meteo_data_future = meteo_data_future.dropna()

### Here we are combining past and future tables
### For any given date, we use the historical data if it is available, otherwise we will use the past days of forecast
meteo_data_historical.insert(0,"type","historical")
meteo_data_future.insert(0,"type","future")
meteo_data_all = pd.concat([meteo_data_historical,meteo_data_future],axis=0)
meteo_data_all["priorty"] = meteo_data_all.groupby("dt")["type"].rank(ascending=False)
meteo_data_all = meteo_data_all[meteo_data_all["priorty"] == 1]
meteo_data_all = meteo_data_all.sort_values("dt")
meteo_data_all = meteo_data_all.drop(["type","priorty"],axis=1)

# Prepare the main data table
### First date in the table is the same with meteorology data.
### The last date in the main table is the day we will predict (d+1)



df_dates = pd.date_range(start_date,last_datetime_to_be_forecasted,freq="1h")
df_dates = df_dates.tz_localize("Europe/Istanbul")

df = pd.DataFrame()
df["dt"] = df_dates

### Adding the price data to main table
df = df.merge(price_data,how="left")

### Adding the meteorology data
df = df.merge(meteo_data_all,how="left")

### We use the last available price data (2 days ago)
### Note: Unlike solar production, prices can be negative.
df["price_lag_2days"] = df["price"].shift(2*24)

### Adding hour of day as a feature (prices have strong hourly patterns)
df["hour"] = df["dt"].dt.hour

### Preparing the train data
### Here dropping rows with NA values will remove the test data in our case
### You can directly filter train data with other ways.
### It is highly recommended to check if the resulting table is as you want
train_X = df.dropna().drop(["dt","price"],axis=1)
train_y = df.dropna()["price"]

print(f"Variables used in model: {train_X.columns.tolist()}")

train_X = sm.add_constant(train_X)

model = sm.OLS(train_y, train_X)
results = model.fit()

### We will forecast last 24 hour in our main table (day d+1)
next_day_X = df.iloc[-24:].drop(["dt","price"],axis=1)
next_day_X = sm.add_constant(next_day_X)

next_day_pred = results.predict(next_day_X)

### Note: Unlike solar production, electricity prices CAN be negative.
### Do NOT clip predictions to zero.

### You should print your results as a python list as the following code
### This should be the last line of your code!
### Since my array is a numpy array, I converted it to list using .tolist()
### If you use python list, you do not need to do that
### An example output with the correct format is as follows:
### [150.2, 142.5, 138.0, 125.3, 118.7, 115.2, 120.8, 145.6, 189.3, 225.7, 248.1, 260.5, 255.8, 245.3, 238.7, 235.2, 240.8, 265.6, 289.3, 275.7, 248.1, 210.5, 185.8, 165.3]
print(next_day_pred.tolist())
