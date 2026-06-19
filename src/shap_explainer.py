import os
import pickle
import pandas as pd
import numpy as np
import shap
import json

from preprocess import preprocess_all
from aggregation import aggregate_to_weekly
from features import create_features
from predict import get_latest_state
from config import DATA_DIR, MODEL_PATH


def categorize_feature(feature_name: str) -> str:
    """
    Categorizes a feature into Spend, Efficiency, or Seasonality drivers.
    """
    f_lower = feature_name.lower()
    if 'spend' in f_lower or 'budget' in f_lower or 'cost' in f_lower:
        return 'Spend Drivers'
    elif 'ctr' in f_lower or 'cvr' in f_lower or 'roas' in f_lower or 'cpa' in f_lower or 'rpc' in f_lower or 'clicks' in f_lower or 'conversions' in f_lower or 'impressions' in f_lower:
        return 'Efficiency Drivers'
    elif 'week' in f_lower or 'month' in f_lower or 'quarter' in f_lower:
        return 'Seasonality Drivers'
    else:
        return 'Efficiency Drivers'  # Default fallback

def compute_shap_explanations(data_dir: str, model_path: str) -> str:
    """
    Calculates aggregate SHAP values for the Revenue and ROAS models 
    over the latest observed state and returns a structured JSON string.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model pickle file not found at {model_path}")
        
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
        
    daily = preprocess_all(data_dir)
    weekly = aggregate_to_weekly(daily)
    feat_df = create_features(weekly, is_training=False)
    
    # We use the latest week's state as the explanation target
    latest_state = get_latest_state(feat_df)
    
    results = {}
    
    # Explain Revenue (using the 30d forecast model as representative)
    if 'revenue_30d' in model_dict:
        model = model_dict['revenue_30d']
        features = model_dict['feature_cols_revenue']['30d']
        
        # Prepare data row
        X_explain = latest_state.copy()
        # Set a baseline planned horizon spend (mean of last 4 weeks of spend * 4)
        baseline_spends = feat_df.groupby('campaign_id', observed=False)['spend'].transform(lambda x: x.tail(4).mean() * 4)
        # Match latest_state indexes
        latest_spends = baseline_spends.loc[latest_state.index]
        X_explain['horizon_spend_30d'] = latest_spends
        
        X = X_explain[features].copy()
        for c in ['campaign_id', 'campaign_type', 'channel']:
            X[c] = X[c].astype('category')
            
        # Drop rows with nulls
        X = X.dropna()
        
        if len(X) > 0:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # For multiclass or single output, handle shape
            # In regression, shap_values shape is (num_rows, num_features)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
                
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            driver_importance = []
            for i, feat_name in enumerate(features):
                imp = float(mean_abs_shap[i])
                category = categorize_feature(feat_name)
                driver_importance.append({
                    'feature': feat_name,
                    'importance': imp,
                    'category': category
                })
                
            # Sort by importance and take top 5
            driver_importance = sorted(driver_importance, key=lambda x: x['importance'], reverse=True)
            top_5_rev = driver_importance[:5]
            
            results['revenue'] = {
                'drivers': top_5_rev,
                'summary': f"Forecast is heavily driven by {top_5_rev[0]['feature']} ({top_5_rev[0]['category']}) and {top_5_rev[1]['feature']} ({top_5_rev[1]['category']})."
            }
        else:
            results['revenue'] = {'drivers': [], 'summary': "No data available."}
            
    # Explain ROAS (using the auxiliary weekly ROAS model)
    if 'aux_roas' in model_dict:
        model = model_dict['aux_roas']
        features = model_dict['feature_cols_roas']
        
        X = latest_state[features].copy()
        for c in ['campaign_id', 'campaign_type', 'channel']:
            X[c] = X[c].astype('category')
            
        X = X.dropna()
        
        if len(X) > 0:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
                
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            driver_importance = []
            for i, feat_name in enumerate(features):
                imp = float(mean_abs_shap[i])
                category = categorize_feature(feat_name)
                driver_importance.append({
                    'feature': feat_name,
                    'importance': imp,
                    'category': category
                })
                
            driver_importance = sorted(driver_importance, key=lambda x: x['importance'], reverse=True)
            top_5_roas = driver_importance[:5]
            
            results['roas'] = {
                'drivers': top_5_roas,
                'summary': f"Efficiency is primarily driven by {top_5_roas[0]['feature']} ({top_5_roas[0]['category']}) and {top_5_roas[1]['feature']} ({top_5_roas[1]['category']})."
            }
        else:
            results['roas'] = {'drivers': [], 'summary': "No data available."}
            
    return json.dumps(results, indent=2)

if __name__ == "__main__":
    drivers_json = compute_shap_explanations(DATA_DIR, MODEL_PATH)
    print(drivers_json)
