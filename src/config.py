"""
Centralized configuration for the PMIE project.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "pickle", "model.pkl")

# ── Forecasting & Horizons ───────────────────────────────────────────
FORECAST_HORIZONS = ['30d', '60d', '90d']
HORIZON_TO_WEEKS = {
    '30d': 4,
    '60d': 8,
    '90d': 13
}

# ── Monte Carlo Settings ─────────────────────────────────────────────
MC_SIMULATIONS = 10000
MC_RANDOM_SEED = 101

# ── Conformal Prediction ─────────────────────────────────────────────
CONFORMAL_COVERAGE = 0.80

# ── Drift Monitoring ─────────────────────────────────────────────────
DRIFT_THRESHOLDS = {
    "stable": 0.10,
    "moderate": 0.25
}

# ── Default File Names ───────────────────────────────────────────────
PREDICTIONS_OUTPUT_FILE = os.path.join(BASE_DIR, "output", "predictions.csv")
