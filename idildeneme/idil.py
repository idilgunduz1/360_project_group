"""
IE360 Electricity Price Forecasting - Final Version
=====================================================
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "ie360_price_project" / "data"
RAW_DIR = DATA_DIR / "raw"

TIME_COL = "timestamp"
TARGET_COL = "mcp"

# Feature parameters
LAG_HOURS = [1, 2, 3, 6, 12, 24, 48, 168]
ROLL_WINDOWS = [6, 24, 168]


def load_price_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load price data from local CSV."""
    file_path = RAW_DIR / "price.csv" if path is None else Path(path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Price data not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
    df = df.dropna(subset=[TIME_COL, TARGET_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features."""
    out = df.copy()
    ts = pd.to_datetime(out[TIME_COL])
    
    out['hour'] = ts.dt.hour
    out['day_of_week'] = ts.dt.dayofweek
    out['day_of_month'] = ts.dt.day
    out['month'] = ts.dt.month
    out['is_weekend'] = (out['day_of_week'] >= 5).astype(int)
    out['is_peak_hour'] = ((out['hour'] >= 8) & (out['hour'] <= 20)).astype(int)
    
    # Cyclical encoding
    out['hour_sin'] = np.sin(2 * np.pi * out['hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * out['hour'] / 24)
    out['dow_sin'] = np.sin(2 * np.pi * out['day_of_week'] / 7)
    out['dow_cos'] = np.cos(2 * np.pi * out['day_of_week'] / 7)
    
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag features."""
    out = df.copy()
    for lag in LAG_HOURS:
        out[f'lag_{lag}'] = out[TARGET_COL].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling statistics."""
    out = df.copy()
    for w in ROLL_WINDOWS:
        out[f'roll_mean_{w}'] = out[TARGET_COL].shift(1).rolling(w, min_periods=1).mean()
        out[f'roll_std_{w}'] = out[TARGET_COL].shift(1).rolling(w, min_periods=1).std().fillna(0)
        out[f'roll_min_{w}'] = out[TARGET_COL].shift(1).rolling(w, min_periods=1).min()
        out[f'roll_max_{w}'] = out[TARGET_COL].shift(1).rolling(w, min_periods=1).max()
    
    # Price changes
    out['price_diff_1'] = out[TARGET_COL].diff(1)
    out['price_diff_24'] = out[TARGET_COL].diff(24)
    
    # Clean up
    out = out.replace([np.inf, -np.inf], 0)
    
    return out


def build_training_table(price_df: pd.DataFrame) -> pd.DataFrame:
    """Build feature table for training."""
    df = price_df.copy()
    
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    
    # Clean up
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        if col != TARGET_COL:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
    
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    
    return df


def build_future_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """Build features for future 24-hour prediction."""
    history = price_df.sort_values(TIME_COL).reset_index(drop=True)
    last_ts = history[TIME_COL].max()
    
    # Future timestamps
    future_index = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=24, freq='h')
    future_df = pd.DataFrame({TIME_COL: future_index})
    future_df[TARGET_COL] = np.nan
    
    # Combine with history
    combined = pd.concat([history[[TIME_COL, TARGET_COL]], future_df], ignore_index=True)
    combined = add_calendar_features(combined)
    combined = add_lag_features(combined)
    combined = add_rolling_features(combined)
    
    # Get future rows
    future_features = combined[combined[TIME_COL].isin(future_index)].copy()
    
    # Clean up
    numeric_cols = future_features.select_dtypes(include=['number']).columns.tolist()
    for col in numeric_cols:
        if col != TARGET_COL:
            future_features[col] = future_features[col].replace([np.inf, -np.inf], np.nan)
            if future_features[col].isna().any():
                hist_median = combined[col].median()
                if pd.isna(hist_median):
                    hist_median = 0
                future_features[col] = future_features[col].fillna(hist_median)
    
    return future_features


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns."""
    exclude = [TIME_COL, TARGET_COL]
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64']]
    return feature_cols


def naive_forecast(price_df: pd.DataFrame, seasonality: int = 24) -> np.ndarray:
    """Simple naive forecast."""
    history = price_df.sort_values(TIME_COL).reset_index(drop=True)
    if len(history) < seasonality:
        return np.full(24, history[TARGET_COL].iloc[-1])
    return history[TARGET_COL].iloc[-seasonality:].values.copy()


def train_model(train_df: pd.DataFrame):
    """Train prediction model."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    feature_cols = get_feature_columns(train_df)
    X = train_df[feature_cols].fillna(train_df[feature_cols].median())
    y = train_df[TARGET_COL]
    
    models = {}
    
    # HistGradientBoosting
    hgb = HistGradientBoostingRegressor(
        max_iter=150,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    hgb.fit(X, y)
    models['hgb'] = hgb
    
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    models['rf'] = rf
    
    # Ridge
    ridge = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0, random_state=42))
    ])
    ridge.fit(X, y)
    models['ridge'] = ridge
    
    return models, feature_cols


def predict_ensemble(models: dict, feature_cols: list, future_df: pd.DataFrame) -> np.ndarray:
    """Make ensemble predictions."""
    X_future = future_df[feature_cols].fillna(future_df[feature_cols].median())
    
    # Weighted ensemble
    preds = (
        0.4 * models['hgb'].predict(X_future) +
        0.4 * models['rf'].predict(X_future) +
        0.2 * models['ridge'].predict(X_future)
    )
    
    return np.asarray(preds, dtype=float)


def main():
    """Main forecasting pipeline."""
    print("[INFO] Loading price data...")
    price_df = load_price_data()
    print(f"[INFO] Loaded {len(price_df)} hours")
    
    print("[INFO] Building features...")
    train_df = build_training_table(price_df)
    future_df = build_future_features(price_df)
    
    print("[INFO] Training model...")
    models, feature_cols = train_model(train_df)
    
    print("[INFO] Making predictions...")
    preds = predict_ensemble(models, feature_cols, future_df)
    
    # Ensure non-negative
    preds = np.maximum(preds, 0)
    
    # Format output as required list
    pred_list = [round(float(p), 2) for p in preds]
    print(f"\n{pred_list}")
    
    return pred_list


if __name__ == "__main__":
    main()