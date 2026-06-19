import numpy as np
import pandas as pd
from config import DATA_DIR
from preprocess import preprocess_all


def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates daily campaign data to campaign-week level (Monday-start).
    """
    df = df.copy()
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate Monday-start of the week
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.weekday, unit='D')
    
    # Aggregation rules
    agg_rules = {
        'spend': 'sum',
        'clicks': 'sum',
        'impressions': 'sum',
        'conversions': 'sum',
        'revenue': 'sum',
        'budget': 'mean',  # Average daily budget for the week
    }
    
    # Non-aggregatable columns we want to retain
    meta_cols = ['campaign_name', 'campaign_type', 'channel', 'exclude_from_modeling']
    
    # First, let's group and aggregate
    grouped = df.groupby(['campaign_id', 'week_start']).agg(agg_rules).reset_index()
    
    # Fetch first occurrences of metadata for each campaign
    meta_df = df.groupby('campaign_id')[meta_cols].first().reset_index()
    
    # Merge metadata back
    weekly_df = pd.merge(grouped, meta_df, on='campaign_id', how='left')
    
    # Rename week_start to date to preserve unified schema
    weekly_df = weekly_df.rename(columns={'week_start': 'date'})
    
    # Reorder columns
    cols = ['date', 'campaign_id', 'campaign_name', 'campaign_type', 'channel', 
            'spend', 'clicks', 'impressions', 'conversions', 'revenue', 'budget', 'exclude_from_modeling']
    
    # Sort chronologically per campaign
    weekly_df = weekly_df[cols].sort_values(['channel', 'campaign_id', 'date']).reset_index(drop=True)
    return weekly_df

def aggregate_predictions_hierarchical(campaign_preds: pd.DataFrame) -> pd.DataFrame:
    """
    Performs bottom-up aggregation of campaign predictions to guarantee reconciliation.
    
    Parameters:
      campaign_preds: DataFrame containing predictions at the campaign level.
        Expected columns: ['date', 'campaign_id', 'campaign_name', 'campaign_type', 'channel', 
                           'spend', 'revenue', 'revenue_p10', 'revenue_p90', ...]
                           
    Returns:
      DataFrame containing reconciled predictions at all levels:
        - Campaign level
        - Campaign Type level
        - Channel level
        - Total level
    """
    df = campaign_preds.copy()
    
    # 1. Campaign Level
    df['level'] = 'campaign'
    
    # 2. Campaign Type Level
    type_grouped = df.groupby(['date', 'channel', 'campaign_type']).agg({
        'spend': 'sum',
        'revenue': 'sum',
        'revenue_p10': 'sum',
        'revenue_p90': 'sum'
    }).reset_index()
    type_grouped['level'] = 'campaign_type'
    type_grouped['campaign_id'] = 'ALL_' + type_grouped['campaign_type'].str.upper()
    type_grouped['campaign_name'] = 'All ' + type_grouped['campaign_type']
    
    # 3. Channel Level
    channel_grouped = df.groupby(['date', 'channel']).agg({
        'spend': 'sum',
        'revenue': 'sum',
        'revenue_p10': 'sum',
        'revenue_p90': 'sum'
    }).reset_index()
    channel_grouped['level'] = 'channel'
    channel_grouped['campaign_type'] = 'ALL'
    channel_grouped['campaign_id'] = 'ALL_' + channel_grouped['channel'].str.upper()
    channel_grouped['campaign_name'] = 'All ' + channel_grouped['channel'].str.capitalize()
    
    # 4. Total Level
    total_grouped = df.groupby(['date']).agg({
        'spend': 'sum',
        'revenue': 'sum',
        'revenue_p10': 'sum',
        'revenue_p90': 'sum'
    }).reset_index()
    total_grouped['level'] = 'total'
    total_grouped['channel'] = 'total'
    total_grouped['campaign_type'] = 'ALL'
    total_grouped['campaign_id'] = 'TOTAL'
    total_grouped['campaign_name'] = 'Total Portfolio'
    
    # Combine all levels
    reconciled = pd.concat([df, type_grouped, channel_grouped, total_grouped], ignore_index=True)
    
    # Derive ROAS for all rows to guarantee mathematical consistency
    # ROAS = Revenue / Spend
    reconciled['roas'] = reconciled['revenue'] / reconciled['spend'].replace(0, np.nan)
    reconciled['roas'] = reconciled['roas'].fillna(0.0)
    
    # Reconciled ROAS quantiles using revenue bounds and spend
    reconciled['roas_p10'] = reconciled['revenue_p10'] / reconciled['spend'].replace(0, np.nan)
    reconciled['roas_p10'] = reconciled['roas_p10'].fillna(0.0)
    
    reconciled['roas_p90'] = reconciled['revenue_p90'] / reconciled['spend'].replace(0, np.nan)
    reconciled['roas_p90'] = reconciled['roas_p90'].fillna(0.0)
    
    return reconciled

if __name__ == "__main__":
    daily = preprocess_all(DATA_DIR)
    weekly = aggregate_to_weekly(daily)
    print(f"Weekly aggregated shape: {weekly.shape}")
    print(weekly.head())
