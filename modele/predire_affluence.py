"""
Predicts availability (available bikes / free docks) at a station, date and hour, using the
models trained by entrainer_modele.py.

Usage:
    python3 predire_affluence.py --station 21 --date 2025-10-15 --hour 8
    python3 predire_affluence.py --name "CARNOT" --date 2025-10-15 --hour 8
"""
import argparse
import functools
import os

import joblib
import pandas as pd

from jours_feries import is_holiday

MODELS_DIR = os.path.join(os.path.dirname(__file__), "modeles")


@functools.lru_cache(maxsize=1)
def load_reference():
    return pd.read_csv(os.path.join(MODELS_DIR, "stations.csv"))


@functools.lru_cache(maxsize=1)
def load_models():
    bikes = joblib.load(os.path.join(MODELS_DIR, "modele_available_bikes.joblib"))
    stands = joblib.load(os.path.join(MODELS_DIR, "modele_available_bike_stands.joblib"))
    return bikes, stands


@functools.lru_cache(maxsize=1)
def load_risk_classifiers():
    bike_risk = joblib.load(os.path.join(MODELS_DIR, "modele_risque_available_bikes.joblib"))
    dock_risk = joblib.load(os.path.join(MODELS_DIR, "modele_risque_available_bike_stands.joblib"))
    return bike_risk, dock_risk


def risk_state(probability, classifier):
    if probability >= classifier["high_risk_threshold"]:
        return "high_risk"
    if probability >= classifier["uncertain_threshold"]:
        return "uncertain"
    return "safe"


def find_station(reference, station=None, name=None):
    if station is not None:
        row = reference[reference["station"] == station]
        if row.empty:
            raise ValueError(
                f"Station {station} is unknown or doesn't have enough historical data to be trained on."
            )
        return row.iloc[0]

    matches = reference[reference["station_name"].str.contains(name, case=False, na=False)]
    if matches.empty:
        raise ValueError(f"No station matches « {name} ».")
    if len(matches) > 1:
        names = ", ".join(f"{r.station} ({r.station_name})" for r in matches.itertuples())
        raise ValueError(f"Several stations match « {name} »: {names}. Specify the number with --station.")
    return matches.iloc[0]


def build_day_features(station_row, date):
    day_of_week = date.weekday()
    common = {
        "station_code": station_row["station_code"],
        "day_of_week": day_of_week,
        "weekend": int(day_of_week >= 5),
        "holiday": int(is_holiday(date)),
        "capacity": station_row["capacity"],
    }
    return pd.DataFrame([{**common, "hour": h} for h in range(24)])


def predict_day(station=None, name=None, date=None):
    """Predicts all 24 hours of a day in a single model call (more efficient than hour by hour)."""
    reference = load_reference()
    station_row = find_station(reference, station=station, name=name)
    bike_model, dock_model = load_models()
    bike_risk_clf, dock_risk_clf = load_risk_classifiers()

    x = build_day_features(station_row, date)
    capacity = float(station_row["capacity"])
    bikes = bike_model["model"].predict(x[bike_model["features"]]).clip(0, capacity)
    docks = dock_model["model"].predict(x[dock_model["features"]]).clip(0, capacity)
    bike_risk_proba = bike_risk_clf["model"].predict_proba(x[bike_risk_clf["features"]])[:, 1]
    dock_risk_proba = dock_risk_clf["model"].predict_proba(x[dock_risk_clf["features"]])[:, 1]

    return {
        "station": int(station_row["station"]),
        "station_name": station_row["station_name"],
        "capacity": int(capacity),
        "date": date.isoformat(),
        "predictions": [
            {
                "hour": h,
                "predicted_available_bikes": round(float(b), 1),
                "predicted_available_bike_stands": round(float(d), 1),
                "bike_shortage_risk": {
                    "probability": round(float(pb), 3),
                    "state": risk_state(pb, bike_risk_clf),
                },
                "dock_shortage_risk": {
                    "probability": round(float(pd_), 3),
                    "state": risk_state(pd_, dock_risk_clf),
                },
            }
            for h, b, d, pb, pd_ in zip(range(24), bikes, docks, bike_risk_proba, dock_risk_proba)
        ],
    }


def predict(station=None, name=None, date=None, hour=None):
    day = predict_day(station=station, name=name, date=date)
    point = day["predictions"][hour]
    return {
        "station": day["station"],
        "station_name": day["station_name"],
        "capacity": day["capacity"],
        "date": day["date"],
        **point,
    }


def main():
    parser = argparse.ArgumentParser(description="Predicts bike availability at a station/date/hour.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--station", type=int, help="Station number")
    group.add_argument("--name", type=str, help="Station name (or part of it)")
    parser.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD format")
    parser.add_argument("--hour", type=int, required=True, choices=range(0, 24), help="Hour (0-23)")
    args = parser.parse_args()

    date = pd.to_datetime(args.date).date()
    try:
        result = predict(station=args.station, name=args.name, date=date, hour=args.hour)
    except ValueError as e:
        raise SystemExit(f"Error: {e}")

    bike_risk = result["bike_shortage_risk"]
    dock_risk = result["dock_shortage_risk"]
    print(f"Station {result['station']} - {result['station_name']} (capacity {result['capacity']})")
    print(f"On {result['date']} at {result['hour']}:00:")
    print(f"  Available bikes (estimate): {result['predicted_available_bikes']} "
          f"[{bike_risk['state']}, {bike_risk['probability']:.0%} shortage risk]")
    print(f"  Free docks (estimate)     : {result['predicted_available_bike_stands']} "
          f"[{dock_risk['state']}, {dock_risk['probability']:.0%} shortage risk]")


if __name__ == "__main__":
    main()
