import pandas as pd
import numpy as np

from config import TIME_COL, LOCAL_SUBMISSION_FILE
from src.utils import safe_clip_predictions

print(f"Min pred: {preds.min():.3f}")
print(f"Max pred: {preds.max():.3f}")
print(f"Num zeros: {(preds == 0).sum()}")

def save_submission(future_df: pd.DataFrame, preds, path=None) -> pd.DataFrame:
    preds = safe_clip_predictions(preds, lower=0.0)

    out = pd.DataFrame({
        TIME_COL: future_df[TIME_COL].values,
        "prediction": preds,
    })

    if len(out) != 24:
        raise ValueError(f"Submission must contain 24 rows, got {len(out)}")

    if out["prediction"].isna().any():
        raise ValueError("Submission contains NaN predictions")

    save_path = LOCAL_SUBMISSION_FILE if path is None else path
    out.to_csv(save_path, index=False)
    print(f"[INFO] Submission saved to: {save_path}")
    return out