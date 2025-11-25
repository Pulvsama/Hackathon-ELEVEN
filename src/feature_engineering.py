
import numpy as np
import pandas as pd
from typing import Sequence, Set, Tuple, List
from pathlib import Path
from preprocessing import *

def add_school_holiday_feature(df: pd.DataFrame, holiday_days: Set[pd.Timestamp]) -> pd.DataFrame:
    df = df.copy()
    if not holiday_days:
        df["is_school_holiday"] = 0
        return df
    dates = df[DT_COL].dt.normalize()
    df["is_school_holiday"] = dates.map(lambda x: 1 if x in holiday_days else 0).astype(int)
    return df

def _add_binary_holiday_flag(df: pd.DataFrame, holiday_days: Set[pd.Timestamp], col_name: str) -> pd.DataFrame:
    df = df.copy()
    if not holiday_days:
        df[col_name] = 0
        return df
    # Normalize to midnight to match the holiday set
    norm_dates = df[DT_COL].dt.normalize()
    df[col_name] = norm_dates.map(lambda x: 1 if x in holiday_days else 0).astype(int)
    return df

def merge_weather(main_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    return main_df.merge(weather_df, on=DT_COL, how="left", suffixes=("", "_wx"))

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df[DT_COL]

    hour = dt.dt.hour.astype(float)                 # 0..23
    minute = dt.dt.minute.astype(float)             # 0..59
    dow = dt.dt.dayofweek.astype(float)             # 0..6  (Mon=0)
    month0 = (dt.dt.month - 1).astype(float)        # 0..11

    two_pi = 2.0 * np.pi

    df["hour_sin"] = np.sin(two_pi * hour / 24.0)
    df["hour_cos"] = np.cos(two_pi * hour / 24.0)

    df["minute_sin"] = np.sin(two_pi * minute / 60.0)
    df["minute_cos"] = np.cos(two_pi * minute / 60.0)

    df["dow_sin"] = np.sin(two_pi * dow / 7.0)
    df["dow_cos"] = np.cos(two_pi * dow / 7.0)

    df["month_sin"] = np.sin(two_pi * month0 / 12.0)
    df["month_cos"] = np.cos(two_pi * month0 / 12.0)

    return df

def select_features(df: pd.DataFrame, feature_names: Sequence[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy()
    # fill missing precipitation/snow if present
    for col in ("rain_1h", "snow_1h"):
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Identify available time features (added later)
    time_feats = [
    "hour_sin", "hour_cos",
    "minute_sin", "minute_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    ]

    # Keep only features present in df
    valid_features = [c for c in feature_names if c in df.columns] + [c for c in time_feats if c in df.columns]

    num_cols: List[str] = []
    cat_cols: List[str] = []

    for col in valid_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        # categorical columns will be handled separately (only the attraction name)
    if CAT_COL in df.columns:
        cat_cols = [CAT_COL]

    X_df = df[num_cols + cat_cols].copy()
    return X_df, num_cols, cat_cols

def time_based_split(df: pd.DataFrame, frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.dropna(subset=[TARGET_COL]).sort_values(DT_COL)
    idx = int(len(df_sorted) * frac)
    return df_sorted.iloc[:idx].copy(), df_sorted.iloc[idx:].copy()

def create_shifted_time_feature(df: pd.DataFrame, hours: int = 2) -> pd.DataFrame:
    X1 = load_frame(TRAIN_CSV)
    result_df = X1[["DATETIME", "CURRENT_WAIT_TIME", "ENTITY_DESCRIPTION_SHORT"]]
    result_df = result_df.drop_duplicates().sort_values("DATETIME").reset_index(drop=True)
    result_df["DATETIME"] = result_df["DATETIME"] - pd.Timedelta(hours=hours)
    result_df.rename(columns={"CURRENT_WAIT_TIME": f"WAIT_TIME_{hours}H_BEFORE"}, inplace=True)
    return df.merge(result_df, on=["DATETIME", "ENTITY_DESCRIPTION_SHORT"], how="left")

def create_difference_time_feature(df: pd.DataFrame, minutes: int = 15) -> pd.DataFrame:
    X1 = load_frame(TRAIN_CSV)
    result_df = X1[["DATETIME", "CURRENT_WAIT_TIME", "ENTITY_DESCRIPTION_SHORT"]]
    result_df = result_df.drop_duplicates().sort_values("DATETIME").reset_index(drop=True)
    result_df_copy = result_df.copy()
    result_df_copy["DATETIME"] = result_df["DATETIME"] - pd.Timedelta(minutes=minutes)
    result_df.rename(columns={"CURRENT_WAIT_TIME": "WAIT_TIME_DIFF_15M"}, inplace=True)
    result_df["WAIT_TIME_DIFF_15M"] = result_df["WAIT_TIME_DIFF_15M"] - result_df_copy["CURRENT_WAIT_TIME"]
    return df.merge(result_df, on=["DATETIME", "ENTITY_DESCRIPTION_SHORT"], how="left")

def add_fr_de_holiday_flags(df: pd.DataFrame, fr_csv: Path, de_csv: Path) -> pd.DataFrame:
    df = df.copy()
    fr_days = _load_holiday_days_from_csv(fr_csv)
    de_days = _load_holiday_days_from_csv(de_csv)
    df = _add_binary_holiday_flag(df, fr_days, "is_school_holiday_fr")
    df = _add_binary_holiday_flag(df, de_days, "is_school_holiday_de")
    return df

__all__ = [
    "load_frame",
    "make_output_frame",
    "add_school_holiday_feature",
    "_add_binary_holiday_flag",
    "merge_weather",
    "add_time_features",
    "select_features",
    "time_based_split",
    "create_shifted_time_feature",
    "create_difference_time_feature",
    "add_fr_de_holiday_flags",
]