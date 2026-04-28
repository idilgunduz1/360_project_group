from src.data_loader import load_price_data, load_weather_data_for_locations
from src.validate import evaluate_naive, rolling_backtest_ridge, rolling_backtest_naive
from config import TIME_COL


print("[INFO] Loading price data...")
price_df = load_price_data()

start_date = price_df[TIME_COL].min().date().isoformat()
end_date = price_df[TIME_COL].max().date().isoformat()

print("[INFO] Loading weather data...")
weather_df = load_weather_data_for_locations(start_date=start_date, end_date=end_date)

print("[INFO] Evaluating naive baseline...")
naive_score = evaluate_naive(price_df)
print(f"Naive WMAPE: {naive_score:.4f}")

print("[INFO] Running rolling naive backtest...")
bt_naive = rolling_backtest_naive(price_df, start_idx=24*14, horizon=24, step=24, seasonality=24)

print(bt_naive)
print()
print(f"Average Rolling Naive WMAPE: {bt_naive['wmape'].mean():.4f}")

print("[INFO] Running rolling ridge backtest...")
bt = rolling_backtest_ridge(price_df, weather_df=weather_df, start_idx=24*14, horizon=24, step=24)

print(bt)
print()
print(f"Average Ridge WMAPE: {bt['wmape'].mean():.4f}")