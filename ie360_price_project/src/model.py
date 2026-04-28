import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import TIME_COL, TARGET_COL, RANDOM_SEED
from src.features import get_feature_columns


def naive_forecast(price_df: pd.DataFrame, seasonality: int = 24) -> np.ndarray:
    history = price_df.sort_values(TIME_COL).reset_index(drop=True)
    if len(history) < seasonality:
        last_val = history[TARGET_COL].iloc[-1]
        return np.repeat(last_val, 24)

    last_block = history[TARGET_COL].iloc[-seasonality:].values
    return last_block.copy()


def train_ridge_model(train_df: pd.DataFrame):
    feature_cols = get_feature_columns(train_df)
    X = train_df[feature_cols]
    y = train_df[TARGET_COL]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ]
    )
    model.fit(X, y)
    return model, feature_cols


def predict_with_model(model, feature_cols, future_df: pd.DataFrame) -> np.ndarray:
    X_future = future_df[feature_cols]
    preds = model.predict(X_future)
    return np.asarray(preds, dtype=float)