"""
Builds the training dataset from the raw readings in collecte_data/dispo_velos.

Only keeps stations with a usable history: because of a pagination bug in
collecte_disponibilite.py (limit=100 with no offset, while there are ~400 stations in
Toulouse), each day only captures a subset of about 100 stations, not always the same ones.
Result: ~100 stations have an almost complete history (200+ days out of 428) and ~275 others
only have a few scattered days of data, unusable for learning an hourly/weekly cycle. We
therefore restrict training to the well-covered stations.
"""
import glob
import os

import pandas as pd

from jours_feries import is_holiday

AVAILABILITY_DIR = os.path.join(os.path.dirname(__file__), "..", "collecte_data", "dispo_velos")
STATIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "collecte_data", "data_stations")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "dataset_entrainement.csv")
OUTPUT_REFERENCE_FILE = os.path.join(os.path.dirname(__file__), "referentiel_stations.csv")

MIN_DAYS_COVERED = 150  # out of 428 available days; clearly separates well/poorly covered stations


def load_raw_readings():
    files = sorted(glob.glob(os.path.join(AVAILABILITY_DIR, "*.csv")))
    print(f"{len(files)} daily files found in {AVAILABILITY_DIR}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["last_update"] = pd.to_datetime(df["last_update"], utc=True).dt.tz_convert("Europe/Paris")
    return df


def filter_well_covered_stations(df):
    days_per_station = df.groupby("number")["last_update"].apply(lambda s: s.dt.date.nunique())
    kept_stations = days_per_station[days_per_station >= MIN_DAYS_COVERED].index
    print(f"{len(kept_stations)} stations kept out of {df['number'].nunique()} "
          f"(>= {MIN_DAYS_COVERED} days of data)")
    return df[df["number"].isin(kept_stations)].copy()


def load_station_reference():
    files = glob.glob(os.path.join(STATIONS_DIR, "*.csv"))
    # File names are in DD-MM-YYYY format: an alphabetical sort does NOT give chronological
    # order, the date has to be parsed to find the most recent snapshot.
    latest_file = max(files, key=lambda f: pd.to_datetime(
        os.path.basename(f).replace(".csv", ""), format="%d-%m-%Y"
    ))
    df = pd.read_csv(latest_file)
    df = df.rename(columns={"station_numbers": "number", "station_names": "station_name"})
    pos = df["station_pos"].str.strip("()").str.split(",", expand=True).astype(float)
    df["lon"], df["lat"] = pos[0], pos[1]
    return df[["number", "station_name", "lon", "lat"]]


def add_temporal_features(df):
    dt = df["last_update"]
    df["date"] = dt.dt.date
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.weekday  # 0=Monday ... 6=Sunday
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["holiday"] = df["date"].apply(is_holiday).astype(int)
    return df


def build_dataset():
    df = load_raw_readings()
    df = filter_well_covered_stations(df)
    df = add_temporal_features(df)

    reference = load_station_reference()
    df = df.merge(reference.drop(columns=["lon", "lat"]), on="number", how="left")

    columns = [
        "number", "station_name", "date", "hour", "day_of_week", "month",
        "day_of_year", "weekend", "holiday", "bike_stands",
        "available_bikes", "available_bike_stands",
    ]
    df = df[columns].rename(columns={"number": "station", "bike_stands": "capacity"})
    df = df.sort_values(["date", "hour", "station"]).reset_index(drop=True)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset written: {OUTPUT_FILE} ({len(df)} rows, {df['station'].nunique()} stations, "
          f"from {df['date'].min()} to {df['date'].max()})")

    # Geographic reference (station, name, position), used by the web interface (map);
    # restricted to the stations kept in the training dataset.
    kept_reference = reference[reference["number"].isin(df["station"])]
    kept_reference = kept_reference.rename(columns={"number": "station"})
    kept_reference.to_csv(OUTPUT_REFERENCE_FILE, index=False)
    print(f"Geographic reference written: {OUTPUT_REFERENCE_FILE} ({len(kept_reference)} stations)")

    return df


if __name__ == "__main__":
    build_dataset()
