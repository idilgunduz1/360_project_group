import numpy as np
import pandas as pd

def ensure_datetime_sorted(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")

    # If timezone-aware, convert to naive datetime
    if pd.api.types.is_datetime64tz_dtype(out[time_col]):
        out[time_col] = out[time_col].dt.tz_localize(None)

    out = out.dropna(subset=[time_col])
    out = out.sort_values(time_col).reset_index(drop=True)
    return out

def check_missing_hours(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    full_range = pd.date_range(df[time_col].min(), df[time_col].max(), freq="h")
    existing = pd.Index(df[time_col])
    missing = full_range.difference(existing)
    return pd.DataFrame({time_col: missing})

def wmape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return np.abs(y_true - y_pred).sum() / denom

def safe_clip_predictions(preds, lower=0.0, upper=None):
    preds = np.asarray(preds, dtype=float)
    preds = np.maximum(preds, lower)
    if upper is not None:
        preds = np.minimum(preds, upper)
    return preds