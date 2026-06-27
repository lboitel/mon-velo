"""
API + web server to predict bike availability at a station/date/hour.
Reuses the models trained in modele/.

Run with:
    cd mon-velo/interface
    python3 -m uvicorn serveur:app --reload
then open http://127.0.0.1:8000
"""
import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "modele"))

from predire_affluence import load_reference, predict_day  # noqa: E402

app = FastAPI(title="Mon Vélo - Affluence Toulouse")


@app.get("/api/stations")
def list_stations():
    return load_reference().to_dict(orient="records")


@app.get("/api/predict_day")
def predict_day_endpoint(station: int, date: str):
    try:
        day = pd.to_datetime(date).date()
    except ValueError:
        raise HTTPException(400, f"Invalid date: {date}")
    try:
        return predict_day(station=station, date=day)
    except ValueError as e:
        raise HTTPException(404, str(e))


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")
