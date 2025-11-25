import optuna
from preprocessing import *
from feature_engineering import *
from train import *


def main() -> None:
    ensure_directories()

    # 1) Load data
    train_df = load_frame(TRAIN_CSV)
    valid_df = load_frame(VALID_CSV)
    weather_df = load_frame(WEATHER_CSV)

    # 2) Merge weather on DATETIME
    train_df = merge_weather(train_df, weather_df)
    valid_df = merge_weather(valid_df, weather_df)

    train_df = add_fr_de_holiday_flags(train_df, SCHOOL_HOLIDAYS_FR_CSV, SCHOOL_HOLIDAYS_DE_CSV)
    valid_df = add_fr_de_holiday_flags(valid_df, SCHOOL_HOLIDAYS_FR_CSV, SCHOOL_HOLIDAYS_DE_CSV)

    # 3) Add time features
    train_df = add_time_features(train_df)
    valid_df = add_time_features(valid_df)

    # 4) Feature selection
    _, num_cols, cat_cols = select_features(train_df, FEATURES_WITH_WEATHER)
    _, _, _ = select_features(valid_df, FEATURES_WITH_WEATHER)

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, train_df, num_cols, cat_cols), n_trials=50)

    best_params = study.best_params
    print("Best parameters:", best_params)
    print("Best RMSE on holdout:", study.best_value)

    # Train a final model using the best params and obtain preprocessors
    rmse_hold, num_imp, ohe, final_model = train_with_params(best_params, train_df, num_cols, cat_cols)
    print(f"[INFO] Hold-out RMSE (80/20 time split): {rmse_hold:.4f}")

    # 6) Refit on full data and predict
    y_pred = refit_and_predict(final_model, train_df, valid_df, num_cols, cat_cols, num_imp, ohe)

    # 7) Build and save submission
    submission = make_output_frame(valid_df, y_pred, META_COLS)
    submission.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Predictions saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()