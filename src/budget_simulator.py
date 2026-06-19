import os
import pickle
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

from preprocess import preprocess_all
from aggregation import aggregate_to_weekly
from features import create_features
from predict import predict_all_horizons
from config import FORECAST_HORIZONS, HORIZON_TO_WEEKS,DATA_DIR, MODEL_PATH


def hill_function(x, Vmax, K, n):
    """
    Classical Hill function: Vmax * (x^n) / (K^n + x^n)
    """
    return Vmax * (x**n) / (K**n + x**n)

def fit_response_curves(df: pd.DataFrame) -> dict:
    """
    Fits response curves for Search, Shopping, and PMax campaign types.
    For each campaign type, it fits a linear slope for the observed range
    and a Hill function for extrapolation.
    """
    # Group by campaign type and week to get spend vs revenue data points
    df_weekly = df.groupby(['campaign_type', 'date'], observed=False).agg({
        'spend': 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    curves = {}
    valid_types = ['SEARCH', 'SHOPPING', 'PMAX']
    
    for c_type in valid_types:
        sub = df_weekly[df_weekly['campaign_type'].str.upper() == c_type]
        if len(sub) < 5:
            # Fallback parameters if not enough data
            curves[c_type] = {
                'slope': 2.0,
                'Vmax': 100000.0,
                'K': 5000.0,
                'n': 1.0,
                'max_observed_spend': 10000.0,
                'fit_success': False
            }
            continue
            
        spends = sub['spend'].values
        revenues = sub['revenue'].values
        
        # Calculate max observed spend
        max_spend = float(np.max(spends))
        
        # Fit linear slope through origin (for observed range)
        # y = slope * x -> slope = sum(x*y) / sum(x^2)
        slope = float(np.sum(spends * revenues) / np.sum(spends**2)) if np.sum(spends**2) > 0 else 1.0
        
        # Fit Hill function (for extrapolation)
        # Initial guesses – must satisfy lower_bound <= p0 <= upper_bound
        # Vmax: asymptotic revenue ceiling (set to max_rev * 3 to allow headroom)
        # K: spend at half-maximum (set to median spend, a good inflection-point estimate)
        # n: Hill coefficient (start at 1 = Michaelis-Menten; stays in [0.5, 3.0])
        Vmax_lower = float(np.max(revenues) * 1.5)   # lower bound for Vmax
        Vmax_upper = float(np.max(revenues) * 15.0)  # upper bound for Vmax
        Vmax_guess = float(np.max(revenues) * 3.0)   # initial guess (strictly within bounds)
        K_guess    = float(np.median(spends)) if np.median(spends) > 0 else float(np.max(spends) * 0.5)
        K_lower    = float(np.max(spends) * 0.05)
        K_upper    = float(np.max(spends) * 10.0)
        n_guess    = 1.0
        
        bounds = (
            [Vmax_lower, K_lower, 0.5],
            [Vmax_upper, K_upper, 3.0]
        )
        
        # Ensure p0 is within bounds
        p0 = [
            np.clip(Vmax_guess, Vmax_lower + 1, Vmax_upper - 1),
            np.clip(K_guess,    K_lower    + 1, K_upper    - 1),
            1.0
        ]
        
        try:
            popt, _ = curve_fit(
                hill_function, 
                spends, 
                revenues, 
                p0=p0, 
                bounds=bounds,
                maxfev=5000
            )
            Vmax, K, n = popt
            fit_success = True
        except Exception:
            # Fallback: use empirically-derived parameters based on the linear slope.
            # We set Vmax = slope * max_spend * 1.5 (reasonable saturation ceiling),
            # K = max_spend (inflection at observed maximum), n=1.
            # This ensures the Hill curve shows mild saturation rather than a flat line.
            Vmax = slope * max_spend * 1.5 if slope > 0 else Vmax_guess
            K    = max_spend
            n    = 1.0
            fit_success = False
            
        curves[c_type] = {
            'slope': slope,
            'Vmax': float(Vmax),
            'K': float(K),
            'n': float(n),
            'max_observed_spend': max_spend,
            'fit_success': fit_success
        }
        
    return curves

def extrapolate_spend_with_response_curve(spend: float, base_revenue: float, c_type: str, curve_dict: dict) -> float:
    """
    Extrapolates revenue for a spend that exceeds the max observed historical spend,
    using the relative response curve scaling.
    """
    c_type_upper = c_type.upper()
    if c_type_upper not in curve_dict:
        # If no curve fitted, assume linear scaling (no saturation)
        return base_revenue
        
    params = curve_dict[c_type_upper]
    max_spend = params['max_observed_spend']
    
    if spend <= max_spend:
        # Dominance of observed data within range (handled natively by ML model)
        return base_revenue
        
    # Extrapolation beyond observed ranges: scale base revenue by the Hill curve ratio
    v_max, k, n = params['Vmax'], params['K'], params['n']
    
    val_at_max = hill_function(max_spend, v_max, k, n)
    val_at_sim = hill_function(spend, v_max, k, n)
    
    ratio = val_at_sim / val_at_max if val_at_max > 0 else 1.0
    return base_revenue * ratio

def run_budget_simulation(data_dir: str, model_path: str, google_change: float = 0.0, meta_change: float = 0.0, bing_change: float = 0.0) -> dict:
    """
    Simulates the effect of budget changes across 30, 60, and 90 day windows,
    incorporating Hill curves for extrapolation beyond observed ranges.
    """
    # 1. Run predictions with model
    pred_df = predict_all_horizons(data_dir, model_path, google_change, meta_change, bing_change)
    
    # 2. Load historical data to fit curves
    daily = preprocess_all(data_dir)
    weekly = aggregate_to_weekly(daily)
    curves = fit_response_curves(weekly)
    
    # 3. Apply Hill Curve extrapolation for Search, Shopping, and PMax campaigns 
    # where the simulated campaign-level spend exceeds the max observed spend.
    # To do this, we need to know the max observed spend *per campaign* or *per campaign type*.
    # Since we fitted curves at the campaign type level, we use the campaign type curve.
    
    simulated_rows = []
    
    for _, row in pred_df.iterrows():
        new_row = row.copy()
        
        # Only apply extrapolation adjustment to campaign level predictions
        if row['level'] == 'campaign':
            c_type = str(row['campaign_type']).upper()
            if c_type in ['SEARCH', 'SHOPPING', 'PMAX']:
                # Max spend is weekly, let's adjust for horizon
                horizon_weeks = HORIZON_TO_WEEKS.get(row['horizon'], 4)
                max_observed_horizon_spend = curves[c_type]['max_observed_spend'] * horizon_weeks
                
                if row['spend'] > max_observed_horizon_spend:
                    # Adjust predictions
                    new_row['revenue'] = extrapolate_spend_with_response_curve(
                        row['spend'], row['revenue'], c_type, curves
                    )
                    # Adjust bounds proportionally
                    new_row['revenue_p10'] = extrapolate_spend_with_response_curve(
                        row['spend'], row['revenue_p10'], c_type, curves
                    )
                    new_row['revenue_p90'] = extrapolate_spend_with_response_curve(
                        row['spend'], row['revenue_p90'], c_type, curves
                    )
                    
        simulated_rows.append(new_row)
        
    sim_df = pd.DataFrame(simulated_rows)
    
    # 4. Re-aggregate upward to ensure reconciliation is preserved after extrapolation adjustments
    # Split by horizon
    reaggregated_horizons = []
    
    for h in FORECAST_HORIZONS:
        h_df = sim_df[(sim_df['horizon'] == h) & (sim_df['level'] == 'campaign')].copy()
        
        # Bottom-up aggregate
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
        
        combined_h = pd.concat([h_df, type_grouped, channel_grouped, total_grouped], ignore_index=True)
        combined_h['horizon'] = h
        reaggregated_horizons.append(combined_h)
        
    final_sim = pd.concat(reaggregated_horizons, ignore_index=True)
    
    # Recalculate ROAS
    final_sim['roas'] = np.where(final_sim['spend'] > 0, final_sim['revenue'] / final_sim['spend'], 0.0)
    final_sim['roas_p10'] = np.where(final_sim['spend'] > 0, final_sim['revenue_p10'] / final_sim['spend'], 0.0)
    final_sim['roas_p90'] = np.where(final_sim['spend'] > 0, final_sim['revenue_p90'] / final_sim['spend'], 0.0)
    
    # Enforce constraints and monotonicity
    final_sim['revenue_p10'] = np.minimum(final_sim['revenue_p10'], final_sim['revenue'])
    final_sim['revenue_p90'] = np.maximum(final_sim['revenue_p90'], final_sim['revenue'])
    final_sim['roas_p10'] = np.minimum(final_sim['roas_p10'], final_sim['roas'])
    final_sim['roas_p90'] = np.maximum(final_sim['roas_p90'], final_sim['roas'])
    
    # Compile response format
    # Group results by horizon for API output
    api_results = {}
    for h in FORECAST_HORIZONS:
        h_sub = final_sim[final_sim['horizon'] == h]
        api_results[h] = h_sub.to_dict(orient='records')
        
    return {
        'predictions': final_sim,
        'curves': curves,
        'api_format': api_results
    }

if __name__ == "__main__":
    sim_res = run_budget_simulation(DATA_DIR, MODEL_PATH, google_change=0.20, meta_change=-0.10, bing_change=0.05)
    print("Extrapolated 30d Total prediction:")
    tot_pred = sim_res['predictions']
    print(tot_pred[(tot_pred['horizon'] == '30d') & (tot_pred['level'] == 'total')])
