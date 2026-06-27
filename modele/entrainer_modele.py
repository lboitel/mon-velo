"""
For each station/date/hour, trains:
  - two regressions (HistGradientBoostingRegressor): the number of available bikes
    (available_bikes) and the number of free docks to return a bike (available_bike_stands).
  - two risk classifiers (HistGradientBoostingClassifier): the probability that the station
    is (almost) out of bikes, or (almost) full, reduced to 3 readable states
    "safe" / "uncertain" / "high_risk".

Temporal split (not random): the last few weeks serve as the test set, to evaluate the model
the way it would actually be used (predicting the future, not interpolating between
neighboring readings from the same day). Once evaluated, each final model is retrained on the
full dataset (train+test) before being saved, so production doesn't miss out on the most
recent weeks.

A "historical mean/frequency (station, day of week, hour)" baseline is computed to check that
the model brings a real improvement over a simple seasonal average.
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, mean_squared_error, roc_auc_score

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_entrainement.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "modeles")
TEST_DAYS = 42  # ~6 weeks of validation, at the end of the collection period

FEATURES = ["station_code", "hour", "day_of_week", "weekend", "holiday", "capacity"]
TARGETS = ["available_bikes", "available_bike_stands"]
SHORTAGE_THRESHOLD = 1  # "at risk" = 1 or 0 bike/dock left
# Note: "month" and "day_of_year" were tested and dropped, as was weather (temperature,
# precipitation via Open-Meteo). With only ~14 months of history, the model uses these
# features to extrapolate the recent trend (e.g. the August lull) rather than a genuine
# seasonal/weather effect, which hurts predictions on the test weeks. The historical
# mean/frequency (station, day of week, hour) already captures most of the usable signal.

TREE_PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=255, random_state=42)


def load_dataset():
    df = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    # HistGradientBoosting* requires categories encoded as contiguous integers < 255; station
    # numbers go up to ~400, so we recode them as 0..n_stations-1.
    df["station_code"] = df["station"].astype("category").cat.codes
    for target in TARGETS:
        df[f"{target}_shortage"] = (df[target] <= SHORTAGE_THRESHOLD).astype(int)
    return df


def temporal_split(df):
    cutoff = df["date"].max() - pd.Timedelta(days=TEST_DAYS)
    train = df[df["date"] <= cutoff].copy()
    test = df[df["date"] > cutoff].copy()
    print(f"Train: {len(train)} rows (up to {train['date'].max().date()})")
    print(f"Test : {len(test)} rows (from {test['date'].min().date()} to {test['date'].max().date()})")
    return train, test


# ---------------------------------------------------------------------------
# Regression: number of bikes / docks
# ---------------------------------------------------------------------------

def predict_mean_baseline(train, test, target):
    fine_means = train.groupby(["station", "day_of_week", "hour"])[target].mean()
    station_means = train.groupby("station")[target].mean()
    global_mean = train[target].mean()

    keys = list(zip(test["station"], test["day_of_week"], test["hour"]))
    preds = fine_means.reindex(keys)
    preds.index = test.index
    missing = preds.isna()
    preds[missing] = test.loc[missing, "station"].map(station_means).fillna(global_mean)
    return preds.values


def evaluate_regression(y_true, y_pred, capacity, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_pct = mae / capacity.mean() * 100
    print(f"  {name:<18} MAE={mae:5.2f}  RMSE={rmse:5.2f}  (MAE = {mae_pct:4.1f}% of average capacity)")


def train_regressor(data, target):
    model = HistGradientBoostingRegressor(categorical_features=["station_code"], **TREE_PARAMS)
    model.fit(data[FEATURES], data[target])
    return model


def build_regressors(df, train, test):
    for target in TARGETS:
        print(f"\n=== Regression: {target} ===")
        baseline_pred = predict_mean_baseline(train, test, target)
        evaluate_regression(test[target], baseline_pred, test["capacity"], "Baseline (historical mean)")

        eval_model = train_regressor(train, target)
        evaluate_regression(test[target], eval_model.predict(test[FEATURES]), test["capacity"], "HistGradientBoosting")

        final_model = train_regressor(df, target)  # retrained on all the data
        path = os.path.join(MODELS_DIR, f"modele_{target}.joblib")
        joblib.dump({"model": final_model, "features": FEATURES}, path)
        print(f"  Model saved: {path}")


# ---------------------------------------------------------------------------
# Classification: shortage risk (0 or 1 bike/dock left)
# ---------------------------------------------------------------------------

def predict_frequency_baseline(train, test, shortage_target):
    return predict_mean_baseline(train, test, shortage_target)  # same logic, binary target


def evaluate_classification(y_true, p_pred, name):
    p = np.clip(p_pred, 1e-3, 1 - 1e-3)
    print(f"  {name:<18} Brier={brier_score_loss(y_true, p):.4f}  "
          f"LogLoss={log_loss(y_true, p):.4f}  AUC={roc_auc_score(y_true, p):.4f}")


def train_classifier(data, shortage_target):
    model = HistGradientBoostingClassifier(categorical_features=["station_code"], **TREE_PARAMS)
    model.fit(data[FEATURES], data[shortage_target])
    return model


def build_risk_classifiers(df, train, test):
    for target in TARGETS:
        shortage_target = f"{target}_shortage"
        print(f"\n=== Shortage risk (<= {SHORTAGE_THRESHOLD}): {target} "
              f"(base rate = {train[shortage_target].mean():.1%}) ===")

        baseline_pred = predict_frequency_baseline(train, test, shortage_target)
        evaluate_classification(test[shortage_target], baseline_pred, "Baseline (frequency)")

        eval_model = train_classifier(train, shortage_target)
        p_eval_train = eval_model.predict_proba(train[FEATURES])[:, 1]
        p_eval_test = eval_model.predict_proba(test[FEATURES])[:, 1]
        evaluate_classification(test[shortage_target], p_eval_test, "HistGradientBoosting")

        # Thresholds for the 3 states, calibrated on the distribution of probabilities
        # predicted on the train set (so they remain valid for new dates) rather than a fixed
        # 0.33/0.66 threshold, which doesn't transfer from one target to another ("docks full"
        # risk is ~3x rarer than "no bikes left", so its predicted probabilities are
        # structurally lower).
        uncertain_threshold, high_risk_threshold = np.quantile(p_eval_train, [0.70, 0.90])
        states = pd.cut(p_eval_test, bins=[-0.01, uncertain_threshold, high_risk_threshold, 1.01],
                         labels=["safe", "uncertain", "high_risk"])
        calibration = pd.DataFrame({"state": states, "actual": test[shortage_target].values})
        summary = calibration.groupby("state", observed=True)["actual"].agg(["mean", "count"])
        print(f"  Thresholds used: uncertain >= {uncertain_threshold:.3f}, high_risk >= {high_risk_threshold:.3f}")
        print(f"  Calibration (actual shortage rate per state):\n{summary.to_string(float_format='%.3f')}")

        final_model = train_classifier(df, shortage_target)  # retrained on all the data
        path = os.path.join(MODELS_DIR, f"modele_risque_{target}.joblib")
        joblib.dump({
            "model": final_model,
            "features": FEATURES,
            "uncertain_threshold": float(uncertain_threshold),
            "high_risk_threshold": float(high_risk_threshold),
        }, path)
        print(f"  Model saved: {path}")


# ---------------------------------------------------------------------------

def build_station_reference(df):
    ref = (
        df[["station", "station_code", "station_name", "capacity", "date"]]
        .sort_values(["station", "date"])
        .drop_duplicates("station", keep="last")
        .drop(columns="date")
        .sort_values("station")
    )
    positions = pd.read_csv(os.path.join(os.path.dirname(__file__), "referentiel_stations.csv"))
    ref = ref.merge(positions[["station", "lon", "lat"]], on="station", how="left")
    ref.to_csv(os.path.join(MODELS_DIR, "stations.csv"), index=False)
    print(f"\nStation reference saved ({len(ref)} stations)")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_dataset()
    train, test = temporal_split(df)

    build_regressors(df, train, test)
    build_risk_classifiers(df, train, test)
    build_station_reference(df)


if __name__ == "__main__":
    main()
