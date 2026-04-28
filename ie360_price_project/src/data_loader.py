import os
import requests
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import timedelta

from dotenv import load_dotenv

from config import (
    PRICE_FILE,
    TIME_COL,
    TARGET_COL,
    WEATHER_LOCATIONS,
    WEATHER_VARS,
    EPIAS_TGT_URL,
    EPIAS_ELECTRICITY_BASE,
    EPIAS_MCP_ENDPOINT,
)
from src.utils import ensure_datetime_sorted, check_missing_hours

load_dotenv()


def load_price_data(path: Optional[str] = None, use_epias_api: bool = True) -> pd.DataFrame:
    """
    Priority:
    1) Try EPIAS API automatically
    2) If that fails, fallback to local CSV
    """
    if use_epias_api:
        try:
            print("[INFO] Trying to load price data from EPIAS API...")
            df = load_price_data_from_epias(days_back=30)
            print(f"[INFO] Loaded {len(df)} rows from EPIAS API.")

            # cache local copy
            PRICE_FILE.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(PRICE_FILE, index=False)
            print(f"[INFO] Cached API data to: {PRICE_FILE}")
        except Exception as e:
            print(f"[WARN] EPIAS API load failed: {e}")
            print("[WARN] Falling back to local CSV...")

    file_path = PRICE_FILE if path is None else path
    df = pd.read_csv(file_path)

    if TIME_COL not in df.columns:
        for c in ["date", "datetime", "ds", "tarih", "timestamp"]:
            if c in df.columns:
                df = df.rename(columns={c: TIME_COL})
                break

    if TARGET_COL not in df.columns:
        for c in ["PTF", "MCP", "price", "fiyat", "mcp"]:
            if c in df.columns:
                df = df.rename(columns={c: TARGET_COL})
                break

    df = df[[TIME_COL, TARGET_COL]].copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, TARGET_COL])
    df = ensure_datetime_sorted(df, TIME_COL)
    df = df.drop_duplicates(subset=[TIME_COL], keep="last").reset_index(drop=True)

    missing = check_missing_hours(df, TIME_COL)
    if len(missing) > 0:
        print(f"[WARN] Missing hourly timestamps found: {len(missing)}")

    return df


def get_epias_tgt() -> str:
    username = os.getenv("EPIAS_USERNAME")
    password = os.getenv("EPIAS_PASSWORD")

    if not username or not password:
        raise ValueError("EPIAS_USERNAME or EPIAS_PASSWORD is missing in .env")

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
    }
    payload = {
        "username": username,
        "password": password,
    }

    resp = requests.post(EPIAS_TGT_URL, data=payload, headers=headers, timeout=60)
    resp.raise_for_status()

    tgt = resp.text.strip()
    if not tgt.startswith("TGT-"):
        raise ValueError(f"Unexpected TGT response: {tgt}")

    return tgt


def fetch_epias_mcp(start_date: str, end_date: str, tgt: str) -> pd.DataFrame:
    """
    Calls EPIAS MCP/PTF listing endpoint.

    Expected endpoint from EPIAS technical docs:
    POST /v1/markets/dam/data/mcp
    """
    url = f"{EPIAS_ELECTRICITY_BASE}{EPIAS_MCP_ENDPOINT}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "TGT": tgt,
    }

    # NOTE:
    # EPİAŞ request DTO field names may vary slightly by version.
    # This payload is the most likely practical form for current MCP list service.
    payload = {
        "startDate": f"{start_date}T00:00:00+03:00",
        "endDate": f"{end_date}T23:00:00+03:00",
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    return normalize_epias_mcp_response(data)


def normalize_epias_mcp_response(data: dict) -> pd.DataFrame:
    """
    EPİAŞ responses sometimes wrap rows under items/body/data/content.
    We try a few common shapes.
    Expected row-level fields in docs include:
    - date
    - hour
    - price
    """
    candidate = None

    if isinstance(data, list):
        candidate = data
    elif isinstance(data, dict):
        for key in ["items", "data", "body", "content", "result"]:
            if key in data and isinstance(data[key], list):
                candidate = data[key]
                break
            if key in data and isinstance(data[key], dict):
                nested = data[key]
                for subkey in ["items", "data", "content"]:
                    if subkey in nested and isinstance(nested[subkey], list):
                        candidate = nested[subkey]
                        break
                if candidate is not None:
                    break

    if candidate is None:
        raise ValueError(f"Could not parse EPIAS MCP response structure: {data}")

    df = pd.DataFrame(candidate)
    if df.empty:
        raise ValueError("EPIAS MCP response is empty")

    # Common documented fields
    # date: 2023-01-01T00:00:00+03:00
    # hour: optional
    # price: TL/MWh
    if "date" in df.columns and "price" in df.columns:
        out = df[["date", "price"]].copy()
        out = out.rename(columns={"date": TIME_COL, "price": TARGET_COL})

    elif "tarih" in df.columns and "fiyat" in df.columns:
        out = df[["tarih", "fiyat"]].copy()
        out = out.rename(columns={"tarih": TIME_COL, "fiyat": TARGET_COL})

    else:
        # try generic mapping
        time_col = None
        price_col = None
        for c in df.columns:
            if c.lower() in ["date", "datetime", "tarih", "timestamp"]:
                time_col = c
            if c.lower() in ["price", "ptf", "mcp", "fiyat"]:
                price_col = c

        if time_col is None or price_col is None:
            raise ValueError(f"Could not identify time/price columns in EPIAS response: {df.columns.tolist()}")

        out = df[[time_col, price_col]].copy()
        out = out.rename(columns={time_col: TIME_COL, price_col: TARGET_COL})

    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out = out.dropna(subset=[TIME_COL, TARGET_COL])
    out = ensure_datetime_sorted(out, TIME_COL)
    out = out.drop_duplicates(subset=[TIME_COL], keep="last").reset_index(drop=True)

    return out


def load_price_data_from_epias(days_back: int = 30) -> pd.DataFrame:
    """
    Pull historical hourly MCP/PTF from EPIAS automatically.
    Default: last ~2 years for local development.
    """
    tgt = get_epias_tgt()

    end_ts = pd.Timestamp.now(tz="Europe/Istanbul").floor("h")
    start_ts = end_ts - pd.Timedelta(days=days_back)

    start_date = start_ts.date().isoformat()
    end_date = end_ts.date().isoformat()

    df = fetch_epias_mcp(start_date=start_date, end_date=end_date, tgt=tgt)

    missing = check_missing_hours(df, TIME_COL)
    if len(missing) > 0:
        print(f"[WARN] Missing hourly timestamps found from EPIAS: {len(missing)}")

    return df


def _open_meteo_request(url: str, params: dict) -> dict:
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def fetch_open_meteo_historical(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    hourly_vars=None,
) -> pd.DataFrame:
    if hourly_vars is None:
        hourly_vars = WEATHER_VARS

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_vars),
        "timezone": "auto",
    }

    data = _open_meteo_request(url, params)

    if "hourly" not in data or data["hourly"] is None:
        return pd.DataFrame(columns=[TIME_COL] + hourly_vars)

    out = pd.DataFrame(data["hourly"])
    out = out.rename(columns={"time": TIME_COL})
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
    out = ensure_datetime_sorted(out, TIME_COL)
    return out


def fetch_open_meteo_forecast(
    lat: float,
    lon: float,
    hourly_vars=None,
    forecast_days: int = 3,
) -> pd.DataFrame:
    if hourly_vars is None:
        hourly_vars = WEATHER_VARS

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "forecast_days": forecast_days,
        "timezone": "auto",
    }

    data = _open_meteo_request(url, params)

    if "hourly" not in data or data["hourly"] is None:
        return pd.DataFrame(columns=[TIME_COL] + hourly_vars)

    out = pd.DataFrame(data["hourly"])
    out = out.rename(columns={"time": TIME_COL})
    out[TIME_COL] = pd.to_datetime(out[TIME_COL], errors="coerce")
    out = ensure_datetime_sorted(out, TIME_COL)
    return out


def load_weather_data_for_locations(
    start_date: str,
    end_date: str,
    locations: Dict[str, Tuple[float, float]] = WEATHER_LOCATIONS,
) -> pd.DataFrame:
    frames = []

    for loc_name, (lat, lon) in locations.items():
        hist = fetch_open_meteo_historical(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date,
        )
        if hist.empty:
            continue

        rename_map = {col: f"{loc_name}_{col}" for col in hist.columns if col != TIME_COL}
        hist = hist.rename(columns=rename_map)
        frames.append(hist)

    if not frames:
        return pd.DataFrame(columns=[TIME_COL])

    merged = frames[0].copy()
    for df in frames[1:]:
        merged = merged.merge(df, on=TIME_COL, how="outer")

    merged = ensure_datetime_sorted(merged, TIME_COL)
    return merged


def load_future_weather_for_locations(
    locations: Dict[str, Tuple[float, float]] = WEATHER_LOCATIONS,
) -> pd.DataFrame:
    frames = []

    for loc_name, (lat, lon) in locations.items():
        fut = fetch_open_meteo_forecast(lat=lat, lon=lon)
        if fut.empty:
            continue

        rename_map = {col: f"{loc_name}_{col}" for col in fut.columns if col != TIME_COL}
        fut = fut.rename(columns=rename_map)
        frames.append(fut)

    if not frames:
        return pd.DataFrame(columns=[TIME_COL])

    merged = frames[0].copy()
    for df in frames[1:]:
        merged = merged.merge(df, on=TIME_COL, how="outer")

    merged = ensure_datetime_sorted(merged, TIME_COL)
    return merged