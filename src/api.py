import os
import shutil
import json
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import uvicorn

from config import DATA_DIR, MODEL_PATH
from logger import get_logger
from preprocess import preprocess_all
from aggregation import aggregate_to_weekly
from predict import predict_all_horizons
from budget_simulator import run_budget_simulation
from shap_explainer import compute_shap_explanations
from monte_carlo import run_portfolio_monte_carlo
from llm_analyst import generate_analyst_report, monitor_drift, detect_anomalies

logger = get_logger(__name__)

app = FastAPI(
    title="Probabilistic Marketing Intelligence Engine (PMIE) API",
    description="Backend forecasting and scenario simulation API for ecommerce marketing agencies.",
    version="1.0"
)

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    google_change: float = 0.0
    meta_change: float = 0.0
    bing_change: float = 0.0

@app.post("/upload", summary="Upload Ads campaign performance datasets")
async def upload_files(
    google_file: UploadFile = File(None),
    meta_file: UploadFile = File(None),
    bing_file: UploadFile = File(None)
):
    """
    Uploads and overwrites the Google, Meta, and/or Bing Ads CSV stats files in the data directory.
    Validates that files are in CSV format.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    uploaded = []
    
    files_mapping = {
        'google_ads_campaign_stats.csv': google_file,
        'meta_ads_campaign_stats.csv': meta_file,
        'bing_campaign_stats.csv': bing_file
    }
    
    for filename, ufile in files_mapping.items():
        if ufile is not None:
            if not ufile.filename.endswith('.csv'):
                logger.warning(f"Invalid file format uploaded: {ufile.filename}")
                raise HTTPException(status_code=400, detail=f"File {ufile.filename} is not a valid CSV file.")
            
            dest_path = os.path.join(DATA_DIR, filename)
            try:
                with open(dest_path, "wb") as buffer:
                    shutil.copyfileobj(ufile.file, buffer)
                uploaded.append(ufile.filename)
                logger.info(f"Successfully uploaded {ufile.filename}")
            except OSError as e:
                logger.error(f"Failed to save {ufile.filename}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Failed to save {ufile.filename}: {str(e)}")
                
    if not uploaded:
        logger.warning("Upload endpoint called with no files.")
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    return {
        "status": "success",
        "message": f"Successfully uploaded files: {', '.join(uploaded)}",
        "details": "Data pipeline caches updated."
    }

@app.get("/forecast", summary="Get baseline predictions for all horizons")
def get_forecast():
    """
    Returns baseline cumulative forecasts (30d, 60d, 90d) for all campaigns,
    reconciled channels, and total portfolio.
    """
    try:
        df_pred = predict_all_horizons(DATA_DIR, MODEL_PATH)
        # Convert to dictionary format grouped by level and channel
        records = df_pred.to_dict(orient='records')
        return {
            "status": "success",
            "forecasts": records
        }
    except Exception as e:
        logger.error(f"Error generating forecasts: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating forecasts. Please check server logs.")

@app.post("/simulate", summary="Simulate budget adjustments and run Monte Carlo")
def simulate_scenario(req: SimulationRequest):
    """
    Runs budget scenario simulation under planned percentage shifts by channel.
    Executes response curves extrapolation and correlated Monte Carlo simulation.
    """
    try:
        # 1. Run simulator (which calls model and applies Hill curves)
        sim_res = run_budget_simulation(
            DATA_DIR, MODEL_PATH, 
            google_change=req.google_change, 
            meta_change=req.meta_change, 
            bing_change=req.bing_change
        )
        
        # 2. Run Monte Carlo simulation for 30d (horizon_weeks=4)
        daily = preprocess_all(DATA_DIR)
        weekly = aggregate_to_weekly(daily)
        mc_res = run_portfolio_monte_carlo(
            weekly, 
            {
                'google': req.google_change, 
                'meta': req.meta_change, 
                'bing': req.bing_change
            }, 
            horizon_weeks=4
        )
        
        # 3. Generate AI insights based on the scenario
        shap_str = compute_shap_explanations(DATA_DIR, MODEL_PATH)
        shap_dict = json.loads(shap_str)
        report = generate_analyst_report(DATA_DIR, MODEL_PATH, sim_res, shap_dict, mc_res)
        
        return {
            "status": "success",
            "simulation": sim_res['api_format'],
            "monte_carlo_30d": mc_res,
            "ai_insights": report
        }
    except Exception as e:
        logger.error(f"Error running simulation scenario: {str(e)}")
        raise HTTPException(status_code=500, detail="Error running simulation. Please check server logs.")

@app.get("/explain", summary="Get SHAP feature importance drivers")
def explain_model():
    """
    Returns the top feature drivers for the Revenue and ROAS models
    categorized by Spend, Efficiency, and Seasonality.
    """
    try:
        drivers_json = compute_shap_explanations(DATA_DIR, MODEL_PATH)
        return {
            "status": "success",
            "shap_explanation": json.loads(drivers_json)
        }
    except Exception as e:
        logger.error(f"Error computing explanations: {str(e)}")
        raise HTTPException(status_code=500, detail="Error computing explanations. Please check server logs.")

@app.get("/anomalies", summary="Monitor anomalies and data drift metrics")
def get_anomalies_and_drift():
    """
    Returns conformal anomaly metrics for the last 90 days
    and Population Stability Index (PSI) drift monitoring checks.
    """
    try:
        drift = monitor_drift(DATA_DIR)
        anoms = detect_anomalies(DATA_DIR, MODEL_PATH)
        return {
            "status": "success",
            "drift_monitoring": drift,
            "recent_anomalies": anoms,
            "anomaly_status": "Stable" if not anoms else f"{len(anoms)} Anomalies Detected"
        }
    except Exception as e:
        logger.error(f"Error monitoring diagnostics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error monitoring diagnostics. Please check server logs.")

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
