# Mon Vélo — Toulouse Bike-Share Availability Prediction

A personal data project that collects ~14 months of real-time availability data from
Toulouse's public bike-share system (VélôToulouse) and uses it to predict, for any station,
date and hour, how likely you are to find a bike to rent or a free dock to return one.

The pipeline covers the full loop: scheduled data collection from the city's open data API,
dataset preparation, model training (with rigorous baseline comparisons), a prediction CLI,
and a small web app with a map and risk indicators.

> **Note on language:** the codebase (variable/function names, comments, API field names) is
> in English. The web interface itself (the pages a user actually looks at) is in French,
> since the product targets French/Toulouse users.

## Table of contents

- [Project structure](#project-structure)
- [Data pipeline](#data-pipeline)
- [Modeling approach](#modeling-approach)
- [Results](#results)
- [Approaches tried and dropped](#approaches-tried-and-dropped)
- [Web interface](#web-interface)
- [API reference](#api-reference)
- [Setup & usage](#setup--usage)
- [Known limitations & possible improvements](#known-limitations--possible-improvements)

## Project structure

```
mon-velo/
├── collecte_data/                  # Data collection (meant to run on a schedule, e.g. cron)
│   ├── collecte_disponibilite.py   # Polls the real-time availability API, writes to temp/
│   ├── traitement_journalier.py    # Merges the day's temp/ snapshots into dispo_velos/
│   ├── referentiel_stations.py     # Fetches the station reference (name, address, position)
│   ├── dispo_velos/                # One CSV per day: station readings (DD-MM-YYYY.csv)
│   ├── data_stations/               # One CSV per day: station reference snapshot
│   └── temp/                       # Scratch space cleared by traitement_journalier.py
│
├── modele/                         # Data prep + model training + prediction
│   ├── preparer_donnees.py         # Builds the training dataset from dispo_velos/
│   ├── jours_feries.py             # French public holiday calculation (no external dep)
│   ├── entrainer_modele.py         # Trains & evaluates all 4 models, saves them
│   ├── predire_affluence.py        # Prediction logic + CLI
│   ├── dataset_entrainement.csv    # Generated: full training dataset (~818k rows)
│   ├── referentiel_stations.csv    # Generated: station name/position reference
│   └── modeles/                    # Generated: trained models + station reference
│       ├── modele_available_bikes.joblib
│       ├── modele_available_bike_stands.joblib
│       ├── modele_risque_available_bikes.joblib
│       ├── modele_risque_available_bike_stands.joblib
│       └── stations.csv
│
└── interface/                      # Web app
    ├── serveur.py                  # FastAPI app: JSON API + serves the static frontend
    └── static/
        ├── index.html
        ├── app.js
        └── style.css
```

## Data pipeline

### Collection

`collecte_disponibilite.py` polls the Toulouse Métropole open data API
(`api-velo-toulouse-temps-reel`) and appends same-day readings to `collecte_data/temp/`.
`traitement_journalier.py` is meant to run once a day (cron) to merge that day's snapshots,
deduplicate, and write one file per day to `collecte_data/dispo_velos/`.

**Known bug:** `collecte_disponibilite.py` queries the API with `limit=100` and no
pagination, while there are roughly 400 stations in Toulouse. `referentiel_stations.py`
*does* paginate correctly (it's used only for the reference data, not the frequent polling).
As a result, each day's collection only captures a subset of about 100 stations — not always
the same ones, seemingly biased toward whichever stations updated most recently when the
request was made.

Measured impact, across the 428 daily files collected from 2024-07-30 to 2025-09-30: 375
distinct stations appear at least once, but only **~100 have a usable history** (200+ days out
of 428). The other ~275 have a median of just 23 days of scattered data — unusable for
learning an hourly/weekly pattern. The training pipeline filters down to these ~100
well-covered stations (`MIN_DAYS_COVERED = 150` in `preparer_donnees.py`). This bug has not
been fixed in the collection script itself; fixing it (adding an offset loop like
`referentiel_stations.py` already does) is one of the listed future improvements.

### Data quality notes

- No missing days: full daily coverage with zero gaps over the 428-day window.
- Readings land roughly once per hour per station (median 23-24 readings/day), with very
  occasional double readings within the same hour (<1% of cases) — not worth deduplicating
  further.
- `bike_stands` (total docks) is often greater than `available_bikes + available_bike_stands`
  (64.7% of rows). This isn't a data error: it reflects docks that are temporarily out of
  service (broken locks, maintenance) — a normal feature of real-world bike-share systems.
- Station reference files (`data_stations/*.csv`) are named `DD-MM-YYYY.csv`. An alphabetical
  sort of these filenames does **not** give chronological order (e.g. `31-08-2024.csv` sorts
  after `01-01-2025.csv`). `preparer_donnees.py` parses the date before picking the most
  recent snapshot — an earlier version of this code had that bug too.

### Dataset preparation

`preparer_donnees.py` (`build_dataset()`):
1. Loads and concatenates all daily files from `dispo_velos/`.
2. Filters to the well-covered stations.
3. Adds temporal features: `hour`, `day_of_week`, `month`, `day_of_year`, `weekend`,
   `holiday` (via `jours_feries.py`, no external package — French Easter-based holidays
   computed with the Meeus/Jones/Butcher algorithm).
4. Joins the station reference (name) from the most recent `data_stations/` snapshot.
5. Writes `dataset_entrainement.csv` (~818k rows, 100 stations, one row per station/reading)
   and a separate `referentiel_stations.csv` (station, name, longitude, latitude) for the
   web map.

## Modeling approach

Two complementary problems are modeled for each of `available_bikes` and
`available_bike_stands`:

1. **Regression** — predict the actual count. `HistGradientBoostingRegressor`
   (scikit-learn), with `station` as a native categorical feature (recoded to small
   contiguous integers, since HistGradientBoosting* requires categories < 255 and raw station
   numbers go up to ~400).
2. **Risk classification** — predict the probability that the station is (almost) out of
   bikes or (almost) full (`count <= 1`), using `HistGradientBoostingClassifier`, then bucket
   the probability into 3 human-readable states: `safe` / `uncertain` / `high_risk`.

Shared design choices:
- **Temporal train/test split** (not random): the last 42 days are held out, so evaluation
  reflects forecasting the future rather than interpolating between neighboring same-day
  readings.
- **Final models are retrained on the full dataset** (train+test) after evaluation, so
  production isn't missing the most recent 6 weeks of signal.
- **Baseline comparison everywhere**: a simple historical mean/frequency grouped by
  `(station, day_of_week, hour)` is always computed and compared against, to make sure any
  added complexity is actually earning its keep.
- Features used: `station` (categorical), `hour`, `day_of_week`, `weekend`, `holiday`,
  `capacity`. (`month` and `day_of_year` were tried and dropped — see below.)

## Results

### Regression (count prediction)

| Target | Baseline MAE | Model MAE | % of average capacity |
|---|---|---|---|
| `available_bikes` | 3.17 | 3.16 | ~16.5% |
| `available_bike_stands` | 3.31 | 3.31 | ~17.3% |

The trained model is essentially **tied with the naive historical-average baseline**. An
"oracle" baseline (the mean computed directly on the test period itself, i.e. unrealistic
perfect hindsight) reaches MAE ≈ 2.18 — meaning a meaningful share of the remaining error is
genuine day-to-day noise that calendar-only features can't capture. Closing that gap would
need a real-time signal (e.g. the last known reading) or other external data (see
[Limitations](#known-limitations--possible-improvements)).

### Risk classification (shortage probability)

This reframing performs much better than the regression, because "is the station nearly
empty/full" is a more structural, predictable event than the exact count:

| Target | Base rate | Baseline AUC | Model AUC | Model Brier |
|---|---|---|---|---|
| Bike shortage (`available_bikes <= 1`) | 26.9% | 0.841 | **0.846** | 0.116 |
| Dock shortage (`available_bike_stands <= 1`) | 7.7% | 0.825 | **0.836** | 0.068 |

Calibration of the 3-state buckets (actual shortage rate observed within each predicted
state), on the held-out test set:

| State | Bike shortage | Dock shortage |
|---|---|---|
| `safe` | 10.0% | 3.5% |
| `uncertain` | 42.1% | 15.7% |
| `high_risk` | 81.7% | 43.6% |

The two targets needed **different probability thresholds** to define the 3 states. A naive
fixed cut (0.33 / 0.66) works fine for bikes but makes `high_risk` almost empty for docks (17
out of ~24k test rows), since dock shortages are ~3.5x rarer and the model's predicted
probabilities for that target are structurally lower. The thresholds are instead calibrated
per target as the 70th/90th percentile of the predicted probability on the training set
(`entrainer_modele.py`), which gives well-separated, reasonably sized buckets for both
targets.

## Approaches tried and dropped

Documented here so they aren't re-tried blindly later.

- **`month` / `day_of_year` as features**: dropped. With only ~14 months of history, the
  model used them to extrapolate the *recent* trend (e.g. the August lull) rather than learn
  a genuine yearly seasonal effect, which hurt accuracy on the test weeks (MAE 3.41 with vs.
  3.16 without, for `available_bikes`).
- **Weather (temperature, precipitation)**: dropped. Hourly historical weather for Toulouse
  was pulled from the free [Open-Meteo](https://open-meteo.com/) archive API (no key
  required) and tested in several forms — raw temperature, raw precipitation, and a binary
  rain flag. None improved on the no-weather baseline (best case: MAE 3.165 vs. 3.162
  without). The raw effect of rain on bike availability in the data is tiny (5.45 vs. 5.65
  average available bikes) and gets lost in noise. Temperature reproduced the same
  recency-bias problem as `month`.
- **More complex regressors**: tested higher `max_leaf_nodes`, more iterations, adding the
  historical-mean as an extra feature, and `RandomForestRegressor` — none meaningfully beat
  the simple historical-average baseline for the count regression task. This is consistent
  with the oracle-ceiling finding above: most of the remaining error is irreducible without a
  new source of signal, not a modeling problem.
- **Fixed 0.33/0.66 thresholds for the 3 risk states**: dropped in favor of per-target
  percentile-based thresholds (see [Results](#results)).

## Web interface

A minimal FastAPI + vanilla JS app (no build step, no frontend framework):

- **Map** (Leaflet, OpenStreetMap tiles) showing all ~100 well-covered stations; click a
  marker or use the search box to select one.
- **Prediction tab** (default view): pick a date and hour (slider), see two color-coded risk
  cards — one for bikes, one for docks — showing just the state (`safe` / `uncertain` /
  `high_risk`), deliberately with **no raw numbers**, to communicate the genuine uncertainty
  honestly rather than implying false precision.
- **Advanced stats tab**: a line chart of the predicted bike/dock counts across all 24 hours
  of the selected day.

## API reference

Base URL: `http://127.0.0.1:8000` (when running `serveur.py` locally).

### `GET /api/stations`

Returns the list of well-covered stations with their reference data.

```json
[
  { "station": 1, "station_code": 0, "station_name": "POIDS DE L'HUILE", "capacity": 19, "lon": 1.445475, "lat": 43.60419 },
  ...
]
```

### `GET /api/predict_day?station={station}&date={YYYY-MM-DD}`

Returns predictions for all 24 hours of the given date at the given station.

```json
{
  "station": 21,
  "station_name": "CARNOT - LABEDA",
  "capacity": 25,
  "date": "2025-10-15",
  "predictions": [
    {
      "hour": 3,
      "predicted_available_bikes": 0.4,
      "predicted_available_bike_stands": 22.8,
      "bike_shortage_risk": { "probability": 0.96, "state": "high_risk" },
      "dock_shortage_risk": { "probability": 0.0, "state": "safe" }
    },
    ...
  ]
}
```

Returns `404` if the station is unknown or not covered by the model, `400` for an invalid date.

## Setup & usage

### Requirements

Python 3.11+, with: `pandas`, `numpy`, `scikit-learn`, `joblib`, `requests`, `fastapi`,
`uvicorn`.

```bash
pip install pandas numpy scikit-learn joblib requests fastapi uvicorn
```

### Rebuild the dataset and retrain all models

```bash
cd modele
python3 preparer_donnees.py   # rebuilds dataset_entrainement.csv + referentiel_stations.csv
python3 entrainer_modele.py   # trains, evaluates, and saves all 4 models into modeles/
```

### Predict from the command line

```bash
cd modele
python3 predire_affluence.py --station 21 --date 2025-10-15 --hour 8
python3 predire_affluence.py --name "CARNOT" --date 2025-10-15 --hour 8
```

### Run the web app

```bash
cd interface
python3 -m uvicorn serveur:app --reload
# open http://127.0.0.1:8000
```

### Run the data collection (intended to run on a schedule)

```bash
cd collecte_data
python3 collecte_disponibilite.py     # run frequently, e.g. hourly, to poll the live API
python3 traitement_journalier.py      # run once a day to merge/dedupe that day's snapshots
python3 referentiel_stations.py       # run periodically to refresh station name/position data
```

## Known limitations & possible improvements

- **Pagination bug** in `collecte_disponibilite.py` limits training data to ~100 of the ~400
  Toulouse stations (see [Data pipeline](#data-pipeline)). Fixing it (pagination like
  `referentiel_stations.py`) would let the model eventually cover the full network, once
  enough history accumulates for the newly-covered stations.
- **Count regression has a measured noise floor** (oracle MAE ≈ 2.18) that calendar-only
  features can't beat. A real-time signal (last known reading, propagated forward) is the
  most promising untested lever, since it directly addresses "what's happening right now"
  rather than "what usually happens at this hour."
- **School holidays** (a calendar feature distinct from public holidays) were never tested
  and could plausibly matter, given how commuting-driven the usage pattern looks.
- The risk-state thresholds (`uncertain`/`high_risk`) are calibrated once at training time on
  a single train/test split; if the system or ridership patterns drift over time, they should
  be recalibrated periodically rather than assumed to stay valid indefinitely.
