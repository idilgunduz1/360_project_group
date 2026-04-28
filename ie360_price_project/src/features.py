import pandas as pd
import numpy as np
import holidays

from config import TIME_COL, TARGET_COL, LAG_HOURS, ROLL_WINDOWS


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out[TIME_COL])

    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["day"] = ts.dt.day
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    tr_holidays = holidays.country_holidays("TR")
    out["is_holiday"] = ts.dt.date.astype("datetime64[ns]").isin(
        pd.to_datetime(list(tr_holidays.keys()))
    ).astype(int)

    # cyclical encoding
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for lag in LAG_HOURS:
        out[f"lag_{lag}"] = out[TARGET_COL].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for w in ROLL_WINDOWS:
        out[f"roll_mean_{w}"] = out[TARGET_COL].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"] = out[TARGET_COL].shift(1).rolling(w).std()
        out[f"roll_min_{w}"] = out[TARGET_COL].shift(1).rolling(w).min()
        out[f"roll_max_{w}"] = out[TARGET_COL].shift(1).rolling(w).max()
    return out


def merge_weather(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    if weather_df is None or weather_df.empty:
        return df.copy()

    out = df.merge(weather_df, on=TIME_COL, how="left")
    return out


def build_training_table(price_df: pd.DataFrame, weather_df: pd.DataFrame = None) -> pd.DataFrame:
    df = price_df.copy()
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = merge_weather(df, weather_df)

    # simple imputation
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if col != TARGET_COL:
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[TARGET_COL])
    df = df.dropna().reset_index(drop=True)
    return df


def build_future_features(price_df: pd.DataFrame, future_weather_df: pd.DataFrame = None) -> pd.DataFrame:
    history = price_df.copy()
    history = history.sort_values(TIME_COL).reset_index(drop=True)

    last_ts = history[TIME_COL].max()
    future_index = pd.date_range(last_ts + pd.Timedelta(hours=25), periods=24, freq="h")

    # NOTE:
    # project setting says when forecasting d+1, you know prices until end of d-1.
    # For local testing, this builder assumes the input history is already truncated correctly.
    future_df = pd.DataFrame({TIME_COL: future_index})
    future_df[TARGET_COL] = np.nan

    # append history + future shell so lag features can be computed
    combined = pd.concat([history[[TIME_COL, TARGET_COL]], future_df], ignore_index=True)
    combined = add_calendar_features(combined)

    # iterative lag construction using available history only
    for lag in [24, 48, 168]:
        combined[f"lag_{lag}"] = combined[TARGET_COL].shift(lag)

    for w in [24, 168]:
        combined[f"roll_mean_{w}"] = combined[TARGET_COL].shift(1).rolling(w).mean()
        combined[f"roll_std_{w}"] = combined[TARGET_COL].shift(1).rolling(w).std()
        combined[f"roll_min_{w}"] = combined[TARGET_COL].shift(1).rolling(w).min()
        combined[f"roll_max_{w}"] = combined[TARGET_COL].shift(1).rolling(w).max()

    combined = merge_weather(combined, future_weather_df)

    future_only = combined[combined[TIME_COL].isin(future_index)].copy()

    numeric_cols = future_only.select_dtypes(include=["number"]).columns.tolist()
    for col in numeric_cols:
        if col != TARGET_COL:
            future_only[col] = future_only[col].fillna(future_only[col].median())

    return future_only.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame):
    exclude = {TIME_COL, TARGET_COL}
    return [c for c in df.columns if c not in exclude]