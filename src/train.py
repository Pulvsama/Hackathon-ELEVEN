from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from typing import List, Tuple, Sequence
from preprocessing import *
from feature_engineering import *


def train_with_params(params: dict, train_df: pd.DataFrame, num_cols: List[str], cat_cols: List[str],
                      target_col=TARGET_COL, cat_feature=CAT_COL):
    # make a shallow copy of params to avoid mutating caller dict
    p = dict(params)

    # time split
    tr_df_split, ho_df_split = time_based_split(train_df, frac=0.8)

    # interaction features
    for df in (tr_df_split, ho_df_split):
        df["is_cold_and_water_attraction"] = (
            (df["feels_like"] < df["feels_like"].mean()) & df[cat_feature].str.contains("Water", case=False)
        ).astype(int)
        df["is_hot_and_water_attraction"] = (
            (df["feels_like"] > df["feels_like"].mean()) & df[cat_feature].str.contains("Water", case=False)
        ).astype(int)

    tr_df_split = create_shifted_time_feature(tr_df_split, hours=1)
    ho_df_split = create_shifted_time_feature(ho_df_split, hours=1)
    tr_df_split = create_shifted_time_feature(tr_df_split, hours=2)
    ho_df_split = create_shifted_time_feature(ho_df_split, hours=2)

    # Add a column "WAIT_TIME_DIFF_15M" that is the difference in waiting time from 15 minutes ago if available
    tr_df_split = create_difference_time_feature(tr_df_split)
    ho_df_split = create_difference_time_feature(ho_df_split)

    num_cols_with_features = num_cols + ["is_cold_and_water_attraction", "is_hot_and_water_attraction","WAIT_TIME_1H_BEFORE", "WAIT_TIME_2H_BEFORE", "WAIT_TIME_DIFF_15M"]

    # numeric preprocessing
    num_imp = SimpleImputer(strategy="median")
    X_tr_num = num_imp.fit_transform(tr_df_split[num_cols_with_features])
    X_ho_num = num_imp.transform(ho_df_split[num_cols_with_features])

    # categorical OHE
    if cat_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_tr_cat = ohe.fit_transform(tr_df_split[cat_cols])
        X_ho_cat = ohe.transform(ho_df_split[cat_cols])
    else:
        ohe = None
        X_tr_cat = np.empty((len(tr_df_split), 0))
        X_ho_cat = np.empty((len(ho_df_split), 0))

    X_tr = np.hstack([X_tr_num, X_tr_cat])
    X_ho = np.hstack([X_ho_num, X_ho_cat])

    y_tr = tr_df_split[target_col].values.astype(float)
    y_ho = ho_df_split[target_col].values.astype(float)


    model = XGBRegressor(**p)

    model.fit(X_tr, y_tr, eval_set=[(X_ho, y_ho)], verbose=False)

    # evaluate
    y_pred_ho = model.predict(X_ho)
    rmse_hold = float(np.sqrt(mean_squared_error(y_ho, y_pred_ho)))

    return rmse_hold, num_imp, ohe, model

def objective(trial, train_df, num_cols, cat_cols, target_col=TARGET_COL, cat_feature=CAT_COL):
    # sample hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 1000, 3000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.03, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "subsample": trial.suggest_float("subsample", 0.5, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 3.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 8, 15),
        "early_stopping_rounds": 200,
        "tree_method": "hist",
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42,
        "eval_metric": "rmse",
    }

    # train and return numeric loss only
    rmse_hold, _, _, _ = train_with_params(params, train_df, num_cols, cat_cols, target_col=target_col, cat_feature=cat_feature)
    return rmse_hold

def refit_and_predict(
    model: object,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    num_imp: SimpleImputer,
    ohe: OneHotEncoder | None
) -> np.ndarray:
    
    train_df = create_shifted_time_feature(train_df, hours=1)
    valid_df = create_shifted_time_feature(valid_df, hours=1)
    train_df = create_shifted_time_feature(train_df, hours=2)
    valid_df = create_shifted_time_feature(valid_df, hours=2)

    # Add a column "WAIT_TIME_DIFF_15M" that is the difference in waiting time from 15 minutes ago if available
    train_df = create_difference_time_feature(train_df)
    valid_df = create_difference_time_feature(valid_df)

    num_cols = num_cols + ["WAIT_TIME_1H_BEFORE", "WAIT_TIME_2H_BEFORE", "WAIT_TIME_DIFF_15M"]

    # Prepare matrices
    X_full_raw = train_df[num_cols + cat_cols].copy()
    y_full = train_df[TARGET_COL].astype(float).values
    X_valid_raw = valid_df[num_cols + cat_cols].copy()

    # Add features that give some interaction between weather and attraction type
    X_full_raw["is_cold_and_water_attraction"] = (
        (X_full_raw["feels_like"] < X_full_raw["feels_like"].mean()) & (X_full_raw[CAT_COL].str.contains("Water", case=False))
    ).astype(int)
    X_valid_raw["is_cold_and_water_attraction"] = (
        (X_valid_raw["feels_like"] < X_valid_raw["feels_like"].mean()) & (X_valid_raw[CAT_COL].str.contains("Water", case=False))
    ).astype(int)
    X_full_raw["is_hot_and_water_attraction"] = (
        (X_full_raw["feels_like"] > X_full_raw["feels_like"].mean()) & (X_full_raw[CAT_COL].str.contains("Water", case=False))
    ).astype(int)
    X_valid_raw["is_hot_and_water_attraction"] = (
        (X_valid_raw["feels_like"] > X_valid_raw["feels_like"].mean()) & (X_valid_raw[CAT_COL].str.contains("Water", case=False))
    ).astype(int)

    num_cols_with_cold = num_cols + ["is_cold_and_water_attraction", "is_hot_and_water_attraction"]
    X_full_num = num_imp.fit_transform(X_full_raw[num_cols_with_cold]) if num_cols_with_cold else np.empty((len(X_full_raw), 0))
    X_valid_num = num_imp.transform(X_valid_raw[num_cols_with_cold]) if num_cols_with_cold else np.empty((len(X_valid_raw), 0))

    if ohe is not None and cat_cols:
        X_full_cat = ohe.fit_transform(X_full_raw[cat_cols])
        X_valid_cat = ohe.transform(X_valid_raw[cat_cols])
    else:
        X_full_cat = np.empty((len(X_full_raw), 0))
        X_valid_cat = np.empty((len(X_valid_raw), 0))

    X_full = np.hstack([X_full_num, X_full_cat])
    X_valid = np.hstack([X_valid_num, X_valid_cat])

    # Determine best_n from early stopping if available
    try:
        booster = model.get_booster()
        best_n = booster.best_ntree_limit
    except Exception:
        best_n = getattr(model, "n_estimators", 1000)
    # Refit new model with the best number of trees
    model_final = XGBRegressor(
        n_estimators=int(best_n),
        learning_rate=0.02,
        max_depth=6,
        subsample=0.7,
        min_child_weight=10,
        colsample_bytree=0.7,
        reg_alpha=0.0,
        reg_lambda=5.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        objective="reg:squarederror",
        eval_metric="rmse",
    )
    model_final.fit(X_full, y_full)
    y_pred = model_final.predict(X_valid)

    return y_pred

__all__ = [
    "train_with_params",
    "objective",
    "refit_and_predict",
]