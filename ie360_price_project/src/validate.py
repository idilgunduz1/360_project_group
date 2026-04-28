import pandas as pd
import numpy as np

from config import TIME_COL, TARGET_COL
from src.features import build_training_table, get_feature_columns
from src.model import naive_forecast, train_ridge_model
from src.utils import wmape


def evaluate_naive(price_df: pd.DataFrame, horizon: int = 24):
    history = price_df.iloc[:-horizon].copy()
    actual = price_df[TARGET_COL].iloc[-horizon:].values
    preds = naive_forecast(history, seasonality=24)
    return wmape(actual, preds)


def rolling_backtest_ridge(
    price_df: pd.DataFrame,
    weather_df: pd.DataFrame = None,
    start_idx: int = 24 * 30,
    horizon: int = 24,
    step: int = 24,
):
    results = []

    for split_end in range(start_idx, len(price_df) - horizon + 1, step):
        train_price = price_df.iloc[:split_end].copy()
        valid_block = price_df.iloc[split_end:split_end + horizon].copy()

        # weather subset if available
        train_weather = None
        valid_weather = None
        if weather_df is not None and not weather_df.empty:
            train_weather = weather_df[weather_df[TIME_COL] <= train_price[TIME_COL].max()].copy()
            valid_weather = weather_df[
                weather_df[TIME_COL].isin(valid_block[TIME_COL])
            ].copy()

        train_df = build_training_table(train_price, train_weather)

        try:
            model, feature_cols = train_ridge_model(train_df)

            # validation features use actual validation timestamps
            valid_feat = valid_block[[TIME_COL, TARGET_COL]].copy()
            from src.features import add_calendar_features, merge_weather

            valid_feat = add_calendar_features(valid_feat)

            # lag features from full combined actual history for honest retrospective validation
            combined = price_df.iloc[:split_end + horizon].copy().reset_index(drop=True)

            for lag in [24, 48, 168]:
                combined[f"lag_{lag}"] = combined[TARGET_COL].shift(lag)

            for w in [24, 168]:
                combined[f"roll_mean_{w}"] = combined[TARGET_COL].shift(1).rolling(w).mean()
                combined[f"roll_std_{w}"] = combined[TARGET_COL].shift(1).rolling(w).std()
                combined[f"roll_min_{w}"] = combined[TARGET_COL].shift(1).rolling(w).min()
                combined[f"roll_max_{w}"] = combined[TARGET_COL].shift(1).rolling(w).max()

            valid_feat = combined[combined[TIME_COL].isin(valid_block[TIME_COL])].copy()
            valid_feat = add_calendar_features(valid_feat[[c for c in valid_feat.columns if c in [TIME_COL, TARGET_COL,
                                                                                                  'lag_24','lag_48','lag_168',
                                                                                                  'roll_mean_24','roll_mean_168',
                                                                                                  'roll_std_24','roll_std_168',
                                                                                                  'roll_min_24','roll_min_168',
                                                                                                  'roll_max_24','roll_max_168']]])
            valid_feat = merge_weather(valid_feat, valid_weather)

            # ensure all required columns exist
            for col in feature_cols:
                if col not in valid_feat.columns:
                    valid_feat[col] = np.nan

            preds = model.predict(valid_feat[feature_cols])
            score = wmape(valid_block[TARGET_COL].values, preds)

        except Exception as e:
            preds = naive_forecast(train_price, seasonality=24)
            score = wmape(valid_block[TARGET_COL].values, preds)

        results.append({
            "train_end": train_price[TIME_COL].max(),
            "valid_start": valid_block[TIME_COL].min(),
            "valid_end": valid_block[TIME_COL].max(),
            "wmape": score,
        })

    return pd.DataFrame(results)


def rolling_backtest_naive(
    price_df,
    start_idx=24 * 30,
    horizon=24,
    step=24,
    seasonality=24,
):
    results = []

    for split_end in range(start_idx, len(price_df) - horizon + 1, step):
        train_price = price_df.iloc[:split_end].copy()
        valid_block = price_df.iloc[split_end:split_end + horizon].copy()

        preds = naive_forecast(train_price, seasonality=seasonality)
        score = wmape(valid_block["mcp"].values, preds)

        results.append({
            "train_end": train_price["timestamp"].max(),
            "valid_start": valid_block["timestamp"].min(),
            "valid_end": valid_block["timestamp"].max(),
            "wmape": score,
        })

    return pd.DataFrame(results)