import os
import pickle
import pandas as pd
import numpy as np

from preprocess import preprocess_all
from aggregation import aggregate_to_weekly
from features import create_features
from conformal import apply_conformal_bounds
from config import HORIZON_TO_WEEKS,DATA_DIR, MODEL_PATH


def get_latest_state(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts the latest record for each campaign, and aligns their dates 
    to the overall maximum date in the dataset to establish a consistent forecast origin.
    """
    df = feat_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    max_date = df['date'].max()
    
    # Sort and group to get the last observed week for each campaign
    latest_rows = df.sort_values('date').groupby('campaign_id', observed=False).last().reset_index()
    latest_rows['date'] = max_date
    return latest_rows

def compute_forecast_confidence(q_rel: float, history: pd.Series) -> float:
    """
    Calculates a forecast confidence percentage based on conformal width, 
    volatility (coefficient of variation), and data volume.
    
    Returns a float between 0.0 and 1.0 (representing 0% to 100%).
    """
    if len(history) <= 4:
        volume_penalty = 0.15
    elif len(history) <= 12:
        volume_penalty = 0.05
    else:
        volume_penalty = 0.0
        
    mean_val = history.mean()
    std_val = history.std()
    
    if mean_val > 0:
        volatility = std_val / mean_val
    else:
        volatility = 0.0
        
    # Cap volatility penalty at 0.20
    volatility_penalty = min(volatility * 0.05, 0.20)
    
    # Base confidence is derived from conformal margin (wider intervals = lower confidence)
    # Since q_rel represents the 80% relative width, higher q_rel means wider bounds.
    base_conf = 1.0 - (q_rel * 0.8)
    
    conf = base_conf - volume_penalty - volatility_penalty
    conf = np.clip(conf, 0.40, 0.98) # Keep within realistic bounds for active campaigns
    return float(conf)

def predict_all_horizons(data_dir: str, model_path: str, google_change: float = 0.0, meta_change: float = 0.0, bing_change: float = 0.0) -> pd.DataFrame:
    """
    Generates reconciled forecasts for 30d, 60d, and 90d horizons, under baseline or simulated adjustments.
    """
    # 1. Load models
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model pickle file not found at {model_path}")
        
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
        
    # 2. Preprocess data
    daily = preprocess_all(data_dir)
    weekly = aggregate_to_weekly(daily)
    
    # We extract weekly history to compute volatility and baseline spends
    # Create features (with is_training=False so we don't need future weeks)
    feat_df = create_features(weekly, is_training=False)
    
    latest_state = get_latest_state(feat_df)
    
    # Calculate baseline weekly spend per campaign as the average of the last 4 observed weeks
    # This prevents single-week spikes/zeros from distorting the forecast horizon spend.
    baseline_spends = {}
    campaign_history = {}
    
    for camp_id in feat_df['campaign_id'].unique():
        camp_sub = feat_df[feat_df['campaign_id'] == camp_id].sort_values('date')
        # Baseline spend is mean of last 4 weeks
        baseline_spends[camp_id] = float(camp_sub['spend'].tail(4).mean())
        # Store revenue history to compute volatility
        campaign_history[camp_id] = camp_sub['revenue']
        
    # Map channel adjustments
    multipliers = {
        'google': 1.0 + google_change,
        'meta': 1.0 + meta_change,
        'bing': 1.0 + bing_change
    }
    
    all_horizon_preds = []
    
    for h_label, weeks in HORIZON_TO_WEEKS.items():
        # Predict for this horizon
        model = model_dict[f'revenue_{h_label}']
        features_all = model_dict['feature_cols_revenue'][h_label]
        q_rel = model_dict.get('conformal_quantiles', {}).get(h_label, 0.25)
        
        h_preds_list = []
        
        for _, row in latest_state.iterrows():
            camp_id = row['campaign_id']
            channel = row['channel']
            exclude = row['exclude_from_modeling']
            
            # Simulated spend for this campaign for 1 week
            m = multipliers.get(channel, 1.0)
            planned_weekly_spend = baseline_spends[camp_id] * m
            
            # Cumulative spend over the horizon
            planned_horizon_spend = planned_weekly_spend * weeks
            
            # Build feature row for model
            feat_row = row.copy()
            feat_row[f'horizon_spend_{h_label}'] = planned_horizon_spend
            
            # Convert to DataFrame row
            X_pred = pd.DataFrame([feat_row[features_all]])
            # Cast categoricals appropriately
            for c in ['campaign_id', 'campaign_type', 'channel']:
                X_pred[c] = X_pred[c].astype('category')
                
            # Model prediction
            if exclude:
                pred_rev = 0.0
            else:
                pred_rev = float(model.predict(X_pred)[0])
                
            # Enforce non-negativity constraint
            pred_rev = max(0.0, pred_rev)
            
            # Conformal bounds
            p10_rev = max(0.0, pred_rev * (1 - q_rel))
            p90_rev = pred_rev * (1 + q_rel)
            
            # Forecast confidence score for this campaign
            hist = campaign_history[camp_id]
            conf_score = compute_forecast_confidence(q_rel, hist)
            
            h_preds_list.append({
                'date': row['date'],
                'campaign_id': camp_id,
                'campaign_name': row['campaign_name'],
                'campaign_type': row['campaign_type'],
                'channel': channel,
                'spend': planned_horizon_spend,
                'revenue': pred_rev,
                'revenue_p50': pred_rev,
                'revenue_p10': p10_rev,
                'revenue_p90': p90_rev,
                'confidence_score': conf_score,
                'level': 'campaign'
            })
            
        h_df = pd.DataFrame(h_preds_list)
        
        # Bottom-up aggregate to reconcile channels, types, and portfolio total
        # Group by levels
        # 1. Campaign Type Level
        type_grouped = h_df.groupby(['date', 'channel', 'campaign_type'], observed=False).agg({
            'spend': 'sum',
            'revenue': 'sum',
            'revenue_p50': 'sum',
            'revenue_p10': 'sum',
            'revenue_p90': 'sum',
            'confidence_score': 'mean'
        }).reset_index()
        type_grouped['level'] = 'campaign_type'
        type_grouped['campaign_id'] = 'ALL_' + type_grouped['campaign_type'].astype(str).str.upper()
        type_grouped['campaign_name'] = 'All ' + type_grouped['campaign_type'].astype(str)
        
        # 2. Channel Level
        channel_grouped = h_df.groupby(['date', 'channel'], observed=False).agg({
            'spend': 'sum',
            'revenue': 'sum',
            'revenue_p50': 'sum',
            'revenue_p10': 'sum',
            'revenue_p90': 'sum',
            'confidence_score': 'mean'
        }).reset_index()
        channel_grouped['level'] = 'channel'
        channel_grouped['campaign_type'] = 'ALL'
        channel_grouped['campaign_id'] = 'ALL_' + channel_grouped['channel'].astype(str).str.upper()
        channel_grouped['campaign_name'] = 'All ' + channel_grouped['channel'].astype(str).str.capitalize()
        
        # 3. Total Level
        total_grouped = h_df.groupby(['date']).agg({
            'spend': 'sum',
            'revenue': 'sum',
            'revenue_p50': 'sum',
            'revenue_p10': 'sum',
            'revenue_p90': 'sum',
            'confidence_score': 'mean'
        }).reset_index()
        total_grouped['level'] = 'total'
        total_grouped['channel'] = 'total'
        total_grouped['campaign_type'] = 'ALL'
        total_grouped['campaign_id'] = 'TOTAL'
        total_grouped['campaign_name'] = 'Total Portfolio'
        
        # Combine
        reconciled_h = pd.concat([h_df, type_grouped, channel_grouped, total_grouped], ignore_index=True)
        
        # Add Horizon label
        reconciled_h['horizon'] = h_label
        all_horizon_preds.append(reconciled_h)
        
    final_reconciled = pd.concat(all_horizon_preds, ignore_index=True)
    
    # Calculate derived ROAS to guarantee consistency
    final_reconciled['roas'] = np.where(final_reconciled['spend'] > 0, final_reconciled['revenue'] / final_reconciled['spend'], 0.0)
    final_reconciled['roas_p10'] = np.where(final_reconciled['spend'] > 0, final_reconciled['revenue_p10'] / final_reconciled['spend'], 0.0)
    final_reconciled['roas_p90'] = np.where(final_reconciled['spend'] > 0, final_reconciled['revenue_p90'] / final_reconciled['spend'], 0.0)
    
    # Enforce constraints
    final_reconciled['roas'] = final_reconciled['roas'].clip(lower=0.0)
    final_reconciled['roas_p10'] = final_reconciled['roas_p10'].clip(lower=0.0)
    final_reconciled['roas_p90'] = final_reconciled['roas_p90'].clip(lower=0.0)
    
    # Force monotonicity of conformal bounds
    final_reconciled['revenue_p10'] = np.minimum(final_reconciled['revenue_p10'], final_reconciled['revenue'])
    final_reconciled['revenue_p90'] = np.maximum(final_reconciled['revenue_p90'], final_reconciled['revenue'])
    final_reconciled['roas_p10'] = np.minimum(final_reconciled['roas_p10'], final_reconciled['roas'])
    final_reconciled['roas_p90'] = np.maximum(final_reconciled['roas_p90'], final_reconciled['roas'])
    
    # Sort and return
    cols = ['level', 'date', 'horizon', 'channel', 'campaign_type', 'campaign_id', 'campaign_name',
            'spend', 'revenue', 'revenue_p50', 'revenue_p10', 'revenue_p90', 'roas', 'roas_p10', 'roas_p90', 'confidence_score']
    return final_reconciled[cols].sort_values(['horizon', 'level', 'channel', 'campaign_id']).reset_index(drop=True)

if __name__ == "__main__":
    df_pred = predict_all_horizons(DATA_DIR, MODEL_PATH)
    print(f"Predictions DataFrame shape: {df_pred.shape}")
    print(df_pred[df_pred['level'] == 'total'])
