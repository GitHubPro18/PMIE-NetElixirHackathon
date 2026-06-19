import pandas as pd
import numpy as np
from preprocess import preprocess_all
from aggregation import aggregate_to_weekly
from config import DATA_DIR


def create_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Computes lags, rolling features, and seasonality markers.
    If is_training is True, it also creates direct horizon targets.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort to ensure chronological order per campaign
    df = df.sort_values(['campaign_id', 'date']).reset_index(drop=True)
    
    # 1. Core Metrics
    df['roas'] = np.where(df['spend'] > 0, df['revenue'] / df['spend'], 0.0)
    df['ctr'] = np.where(df['impressions'] > 0, df['clicks'] / df['impressions'], 0.0)
    df['cvr'] = np.where(df['clicks'] > 0, df['conversions'] / df['clicks'], 0.0)
    df['cpa'] = np.where(df['conversions'] > 0, df['spend'] / df['conversions'], 0.0)
    df['rpc'] = np.where(df['clicks'] > 0, df['revenue'] / df['clicks'], 0.0)
    
    # Ensure business constraints on metrics
    df['ctr'] = df['ctr'].clip(upper=1.0)
    df['cvr'] = df['cvr'].clip(upper=1.0)
    df['roas'] = df['roas'].clip(lower=0.0)
    df['revenue'] = df['revenue'].clip(lower=0.0)
    df['spend'] = df['spend'].clip(lower=0.0)
    
    # 2. Seasonality (extracted from date at week t)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month.astype(int)
    df['quarter'] = df['date'].dt.quarter.astype(int)
    
    # Categoricals as pandas categories for native LightGBM handling
    df['campaign_id'] = df['campaign_id'].astype('category')
    df['campaign_type'] = df['campaign_type'].astype('category')
    df['channel'] = df['channel'].astype('category')
    
    # 3. Lag Features
    lag_cols = ['spend', 'revenue', 'clicks', 'impressions', 'conversions', 'roas']
    for col in lag_cols:
        for lag in [1, 2, 4, 8]:
            df[f'{col}_lag_{lag}'] = df.groupby('campaign_id')[col].shift(lag)
            
    # 4. Rolling Features (Rolling means of last 4, 8, and 12 weeks, including current week)
    # Note: we group by campaign and use rolling mean. To prevent lookahead bias, we can include the current week's metric.
    for col in lag_cols:
        for window in [4, 8, 12]:
            df[f'{col}_rolling_{window}w'] = df.groupby('campaign_id')[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
    # 5. Direct Horizon Targets (Cumulative Future Revenue & Spend)
    # For week t, target is cumulative metric from week t+1 to t+k.
    if is_training:
        horizons = {
            '30d': 4,
            '60d': 8,
            '90d': 13
        }
        for label, weeks in horizons.items():
            # For week t, target = sum(revenue_{t+1} ... revenue_{t+weeks})
            # Achieved by: shift(-weeks) then rolling(weeks).sum()
            # e.g. weeks=4 => shift(-4).rolling(4).sum() gives sum(t+1, t+2, t+3, t+4)
            df[f'target_revenue_{label}'] = df.groupby('campaign_id')['revenue'].transform(
                lambda x: x.shift(-weeks).rolling(window=weeks, min_periods=weeks).sum()
            )
            
            # Future cumulative spend
            df[f'horizon_spend_{label}'] = df.groupby('campaign_id')['spend'].transform(
                lambda x: x.shift(-weeks).rolling(window=weeks, min_periods=weeks).sum()
            )
            
    return df

if __name__ == "__main__":
    
    daily = preprocess_all(DATA_DIR)
    weekly = aggregate_to_weekly(daily)
    feat_df = create_features(weekly, is_training=True)
    
    print(f"Features DataFrame shape: {feat_df.shape}")
    print("Columns:", [c for c in feat_df.columns if 'lag' in c or 'rolling' in c or 'target' in c or 'horizon' in c][:10])
