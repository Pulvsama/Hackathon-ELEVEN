from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Sequence, Set, Tuple, List
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]  # dossier du script
INITIAL_DATA_DIR = REPO_ROOT / "Initial Data"
ADDITIONAL_DATA_DIR = REPO_ROOT / "Additional Data"
OUT_DIR = REPO_ROOT / "Outputs"

TRAIN_CSV = INITIAL_DATA_DIR / "waiting_times_train.csv"
VALID_CSV = INITIAL_DATA_DIR / "waiting_times_X_test_val.csv"
FINAL_CSV = INITIAL_DATA_DIR / "waiting_times_X_test_final.csv"
WEATHER_CSV = INITIAL_DATA_DIR / "weather_data.csv"
OUTPUT_CSV = OUT_DIR / "predictions_validation.csv"

TARGET_COL = "WAIT_TIME_IN_2H"
DT_COL = "DATETIME"
CAT_COL = "ENTITY_DESCRIPTION_SHORT"
SCHOOL_HOLIDAYS_FR_CSV = ADDITIONAL_DATA_DIR / "school_holidays_fr.csv"
SCHOOL_HOLIDAYS_DE_CSV = ADDITIONAL_DATA_DIR / "school_holidays_de.csv"

FEATURES: Sequence[str] = (
    "ADJUST_CAPACITY",
    "DOWNTIME",
    "CURRENT_WAIT_TIME",
    "TIME_TO_PARADE_1",
    "TIME_TO_PARADE_2",
    "TIME_TO_NIGHT_SHOW",
)

FEATURES_WITH_WEATHER: Sequence[str] = (
    "ADJUST_CAPACITY",
    "DOWNTIME",
    "CURRENT_WAIT_TIME",
    "TIME_TO_PARADE_1",
    "TIME_TO_PARADE_2",
    "TIME_TO_NIGHT_SHOW",
    "temp",
    "dew_point",
    "feels_like",
    "pressure",
    "humidity",
    "wind_speed",
    "rain_1h",
    "snow_1h",
    "clouds_all"
)

META_COLS = ("DATETIME", "ENTITY_DESCRIPTION_SHORT")



def load_frame(path: Path, *, datetime_col: str = "DATETIME") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[datetime_col])
    return df



def make_output_frame(valid_df: pd.DataFrame, y_pred, meta_cols: Sequence[str]) -> pd.DataFrame:
    out = (
        valid_df.loc[:, meta_cols].copy()
        .assign(y_pred=y_pred, KEY="Validation")
        .loc[:, list(meta_cols) + ["y_pred", "KEY"]]
        .sort_values(by=meta_cols[0])  
        .reset_index(drop=True)
    )
    return out

def add_weather(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(weather_df, on="DATETIME", how="left")
    return df

def ensure_directories() -> None:
    INITIAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ADDITIONAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_school_holiday_dates(path: Path, zone: str = "C") -> Set[pd.Timestamp]:
    if not path.exists():
        return set()

    df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    if "zone" in df.columns:
        df = df[df["zone"].astype(str).str.upper() == zone.upper()].copy()

    holidays: Set[pd.Timestamp] = set()
    for _, row in df.iterrows():
        start = row["start_date"].normalize()
        end = row["end_date"].normalize()
        cur = start
        while cur < end:
            holidays.add(cur)
            cur += pd.Timedelta(days=1)
    return holidays

def _load_holiday_days_from_csv(csv_path: Path, zone: str | None = None) -> Set[pd.Timestamp]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path, parse_dates=["start_date", "end_date"])
    if zone is not None and "zone" in df.columns:
        df = df[df["zone"].astype(str).str.upper() == zone.upper()].copy()
    days: Set[pd.Timestamp] = set()
    for _, row in df.iterrows():
        # normalize to midnight for consistency
        start = pd.Timestamp(row["start_date"]).normalize()
        end = pd.Timestamp(row["end_date"]).normalize()
        cur = start
        while cur < end:
            days.add(cur)
            cur += pd.Timedelta(days=1)
    return days

__all__ = [
    "REPO_ROOT",
    "INITIAL_DATA_DIR",
    "ADDITIONAL_DATA_DIR",
    "OUT_DIR",
    "TRAIN_CSV",
    "VALID_CSV",
    "FINAL_CSV",
    "WEATHER_CSV",
    "OUTPUT_CSV",
    "TARGET_COL",
    "DT_COL",
    "CAT_COL",
    "SCHOOL_HOLIDAYS_FR_CSV",
    "SCHOOL_HOLIDAYS_DE_CSV",
    "FEATURES",
    "FEATURES_WITH_WEATHER",
    "META_COLS",
    "load_frame",
    "make_output_frame",
    "add_weather",
    "ensure_directories",
    "load_school_holiday_dates",
    "_load_holiday_days_from_csv",
]