from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"

TIME_COL = "timestamp"
TARGET_COL = "mcp"

PRICE_FILE = RAW_DIR / "price.csv"

WEATHER_LOCATIONS = {
    "istanbul": (41.0082, 28.9784),
    "ankara": (39.9334, 32.8597),
    "izmir": (38.4237, 27.1428),
    "bursa": (40.1885, 29.0610),
    "gaziantep": (37.0662, 37.3833),
}

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
]

LAG_HOURS = [24, 48, 168]
ROLL_WINDOWS = [24, 168]

RANDOM_SEED = 42
USE_LOG1P_TARGET = False
LOCAL_SUBMISSION_FILE = OUTPUT_DIR / "submission_local.csv"

# EPIAS
EPIAS_TGT_URL = "https://giris.epias.com.tr/cas/v1/tickets"
EPIAS_ELECTRICITY_BASE = "https://seffaflik.epias.com.tr/electricity-service"
EPIAS_MCP_ENDPOINT = "/v1/markets/dam/data/mcp"