import os
import pickle
import pandas as pd
import numpy as np

from preprocess import preprocess_all
from aggregation import aggregate_to_weekly
from features import create_features
from features import create_features
from train_revenue import get_feature_lists
from config import DATA_DIR, MODEL_PATH

def compute_conformal_margins():
    """
    Computes conformal error quantiles on a calibration split.
    Saves the quantiles in pickle/model.pkl.
    """
    print("Loading data for conformal calibration...")
    daily = preprocess_all(DATA_DIR)
    weekly = aggregate_to_weekly(daily)
    feat_df = create_features(weekly, is_training=True)
    
    # Load trained models
    pickle_path = MODEL_PATH
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Model file not found at {pickle_path}. Please run train_revenue.py first.")
        
    with open(pickle_path, 'rb') as f:
        model_dict = pickle.load(f)
        
    # Split calibration set (e.g. final 15% of chronological data)
    feat_df = feat_df.sort_values('date').reset_index(drop=True)
    unique_dates = feat_df['date'].unique()
    num_dates = len(unique_dates)
    cal_size = int(num_dates * 0.15)
    cal_dates = unique_dates[-cal_size:]
    
    cal_df = feat_df[feat_df['date'].isin(cal_dates)].copy()
    
    conformal_quantiles = {}
    
    for h in ['30d', '60d', '90d']:
        model_key = f'revenue_{h}'
        if model_key not in model_dict:
            print(f"Warning: {model_key} model not found in pickle. Skipping.")
            continue
            
        model = model_dict[model_key]
        features_all = model_dict['feature_cols_revenue'][h]
        target_col = f'target_revenue_{h}'
        
        # Keep non-excluded records with non-null values
        cal_sub = cal_df[~cal_df['exclude_from_modeling']].copy()
        cal_sub = cal_sub.dropna(subset=[target_col] + features_all)
        
        if len(cal_sub) == 0:
            print(f"No calibration samples for horizon {h}")
            conformal_quantiles[h] = 0.25  # Fallback relative error quantile
            continue
            
        # Predict on calibration set
        preds = model.predict(cal_sub[features_all])
        preds = np.clip(preds, 0, None)
        actuals = cal_sub[target_col].values
        
        # Calculate relative errors: |actual - predicted| / max(predicted, epsilon)
        # FIX (Priority 2 – Conformal Coverage):
        # The original formula used (preds + 1.0) as the denominator. When predictions
        # are large (e.g. $50K-$200K/week), adding 1.0 makes negligible difference,
        # so the residual distribution was measured correctly. However, the coverage
        # audit showed the 60d horizon was only 67.5% (target: >=70%). The root cause
        # is that the calibration set is small (cal_size=15% of unique dates) and the
        # in-sample calibration distribution is narrower than out-of-sample prediction
        # errors. We fix this by:
        # (a) using a pure relative error with a small epsilon guard (avoids +1 bias)
        # (b) targeting the 85th percentile instead of 80th to account for the
        #     calibration-to-prediction covariate shift typical in time-series splits.
        eps = max(float(np.median(np.abs(preds))), 1.0)
        rel_errors = np.abs(actuals - preds) / (np.abs(preds) + eps * 0.01)
        
        # Conformal quantile for ~80% coverage (85th percentile of calibration errors
        # accounts for calibration-to-prediction distribution shift)
        q_rel = np.percentile(rel_errors, 85)
        
        # Floor/cap the relative error quantile to realistic boundaries (e.g. between 5% and 100%)
        q_rel = np.clip(q_rel, 0.05, 1.0)
        conformal_quantiles[h] = float(q_rel)
        
        print(f"Horizon {h} - Conformal 80% Relative Error Quantile: {q_rel:.3f} ({q_rel*100:.1f}%)")
        
    # Update pickle
    model_dict['conformal_quantiles'] = conformal_quantiles
    with open(pickle_path, 'wb') as f:
        pickle.dump(model_dict, f)
        
    print(f"Conformal quantiles successfully saved to {pickle_path}!")

def apply_conformal_bounds(preds: pd.Series, horizon: str, model_dict: dict) -> tuple:
    """
    Applies conformal error margins to predictions, yielding P10, P50, and P90 series.
    Enforces business constraints (revenue >= 0) and monotonic intervals (P10 <= P50 <= P90).
    """
    conformal_quantiles = model_dict.get('conformal_quantiles', {'30d': 0.20, '60d': 0.25, '90d': 0.30})
    q_rel = conformal_quantiles.get(horizon, 0.25)
    
    p50 = np.clip(preds, 0, None)
    p10 = np.clip(p50 * (1 - q_rel), 0, None)
    p90 = p50 * (1 + q_rel)
    
    return p10, p50, p90

if __name__ == "__main__":
    compute_conformal_margins()
