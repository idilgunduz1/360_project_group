from config import TIME_COL, TARGET_COL
from src.data_loader import (
    load_price_data,
    load_weather_data_for_locations,
    load_future_weather_for_locations,
)
from src.features import build_training_table, build_future_features
from src.model import train_ridge_model, predict_with_model, naive_forecast
from src.predict import save_submission


def main():
    print("[INFO] Loading price data...")
    price_df = load_price_data()

    start_date = price_df[TIME_COL].min().date().isoformat()
    end_date = price_df[TIME_COL].max().date().isoformat()

    print("[INFO] Loading historical weather...")
    try:
        weather_hist = load_weather_data_for_locations(start_date=start_date, end_date=end_date)
    except Exception as e:
        print(f"[WARN] Historical weather failed: {e}")
        weather_hist = None

    print("[INFO] Building training table...")
    train_df = build_training_table(price_df, weather_hist)

    print("[INFO] Loading future weather...")
    try:
        weather_future = load_future_weather_for_locations()
    except Exception as e:
        print(f"[WARN] Future weather failed: {e}")
        weather_future = None

    print("[INFO] Building future features...")
    future_df = build_future_features(price_df, weather_future)

    try:
        print("[INFO] Training ridge model...")
        model, feature_cols = train_ridge_model(train_df)

        print("[INFO] Predicting next 24 hours...")
        preds = predict_with_model(model, feature_cols, future_df)

    except Exception as e:
        print(f"[WARN] Model failed, switching to naive fallback: {e}")
        preds = naive_forecast(price_df)

    print("[INFO] Saving submission...")
    out = save_submission(future_df, preds)

    print(out.head())
    print("[INFO] Done.")


if __name__ == "__main__":
    main()