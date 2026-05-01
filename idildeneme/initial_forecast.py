"""
IE360 Electricity Price Forecasting - Initial Code
====================================================
Simple baseline forecasting approach
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime, timedelta

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "price.csv"
OUTPUT_FILE = BASE_DIR / "submission.csv"

TIME_COL = "timestamp"
TARGET_COL = "mcp"


def load_price_data() -> pd.DataFrame:
    """Load price data from CSV."""
    df = pd.read_csv(DATA_FILE)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL, TARGET_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    df = df.copy()
    df['hour'] = df[TIME_COL].dt.hour
    df['day_of_week'] = df[TIME_COL].dt.dayofweek
    df['day'] = df[TIME_COL].dt.day
    return df


def seasonal_naive_forecast(train_df: pd.DataFrame, forecast_hours: int = 24) -> np.ndarray:
    """
    Seasonal naive forecast - use same hour from previous days.
    """
    # Get last week data for seasonal patterns
    last_week = train_df.tail(24 * 7)
    
    # Use average of same hour from last week
    forecasts = []
    for h in range(forecast_hours):
        hour_data = last_week[last_week['hour'] == h][TARGET_COL]
        if len(hour_data) > 0:
            forecasts.append(hour_data.mean())
        else:
            # Fallback to overall mean
            forecasts.append(train_df[TARGET_COL].mean())
    
    return np.array(forecasts)


def main():
    print("[INFO] Loading price data...")
    df = load_price_data()
    print(f"[INFO] Loaded {len(df)} rows of price data")
    print(f"[INFO] Date range: {df[TIME_COL].min()} to {df[TIME_COL].max()}")
    
    # Add time features
    df = add_time_features(df)
    
    # Generate forecast
    print("[INFO] Generating forecast...")
    forecasts = seasonal_naive_forecast(df, forecast_hours=24)
    
    # Create submission
    print("[INFO] Creating submission file...")
    submission = pd.DataFrame({
        'timestamp': [df[TIME_COL].max() + timedelta(hours=i+1) for i in range(24)],
        'mcp': forecasts
    })
    
    # Save submission
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Saved forecast to {OUTPUT_FILE}")
    
    # Print forecast summary
    print(f"\n[INFO] Forecast summary:")
    print(f"  Min: {forecasts.min():.2f}")
    print(f"  Max: {forecasts.max():.2f}")
    print(f"  Mean: {forecasts.mean():.2f}")
    print(f"  Std: {forecasts.std():.2f}")
    
    return submission


if __name__ == "__main__":
    main()