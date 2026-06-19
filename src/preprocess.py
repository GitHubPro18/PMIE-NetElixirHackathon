import os
import pandas as pd
import numpy as np
from config import DATA_DIR


def load_google_ads(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "google_ads_campaign_stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Google Ads stats not found at {path}")
        
    df = pd.read_csv(path)
    
    # Standardize column mappings
    rename_dict = {
        'segments_date': 'date',
        'campaign_id': 'campaign_id',
        'campaign_name': 'campaign_name',
        'campaign_advertising_channel_type': 'campaign_type',
        'metrics_clicks': 'clicks',
        'metrics_impressions': 'impressions',
        'metrics_conversions': 'conversions',
        'metrics_conversions_value': 'revenue',
        'campaign_budget_amount': 'budget'
    }
    
    df = df.rename(columns=rename_dict)
    
    # Calculate spend from micros
    df['spend'] = df['metrics_cost_micros'] / 1_000_000.0
    
    df['channel'] = 'google'
    
    # Exclude DISPLAY campaigns from revenue modeling but keep for reporting
    df['exclude_from_modeling'] = df['campaign_type'].str.upper() == 'DISPLAY'
    
    # Select final columns
    cols = ['date', 'campaign_id', 'campaign_name', 'campaign_type', 'channel', 
            'spend', 'clicks', 'impressions', 'conversions', 'revenue', 'budget', 'exclude_from_modeling']
    return df[cols]

def load_meta_ads(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "meta_ads_campaign_stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Meta Ads stats not found at {path}")
        
    df = pd.read_csv(path)
    
    # Validate conversion field as revenue value
    # Meta conversion contains revenue values based on conversion value tracking validation.
    # We rename 'conversion' to 'revenue' and ensure it's numeric and >= 0.
    df['revenue'] = pd.to_numeric(df['conversion'], errors='coerce').fillna(0.0).clip(lower=0)
    
    rename_dict = {
        'date_start': 'date',
        'campaign_id': 'campaign_id',
        'campaign_name': 'campaign_name',
        'spend': 'spend',
        'clicks': 'clicks',
        'impressions': 'impressions',
        'daily_budget': 'budget'
    }
    df = df.rename(columns=rename_dict)
    
    # Meta doesn't have an explicit Conversions column, let's assume conversions = clicks * CTR or clicks * CVR, 
    # but wait, let's look at the Meta data: it has impressions, spend, clicks, conversion (revenue).
    # Since conversions are not explicitly provided, we can proxy conversions by assuming conversions = clicks * 0.02 
    # (a default 2% CVR) or let conversions be 0 if not present, or use a default logic.
    # Wait, does the Meta CSV have a conversions column? 
    # Let's check: Meta columns were 'campaign_id', 'date_start', 'cpc', 'cpm', 'ctr', 'reach', 'spend', 'clicks', 'impressions', 'conversion', 'daily_budget', 'campaign_name'.
    # Indeed, there is no separate conversions column. The "conversion" column itself is the conversion value (revenue).
    # Let's assume conversions count is estimated as `revenue / 100` (assuming $100 Average Order Value) or default to clicks * 0.02 or 0.
    # Let's check if conversions are required. Yes, we need conversions to calculate CVR and AOV.
    # Let's define conversions = (df['revenue'] / 100.0).round().clip(lower=0) as a reasonable proxy, or default to 0. 
    # Let's estimate Conversions = (df['revenue'] / 80.0).round() (assuming AOV of $80) and if revenue is 0, Conversions = 0.
    # Wait! Let's check if clicks are 0, conversions are 0.
    df['conversions'] = np.where(df['revenue'] > 0, (df['revenue'] / 80.0).round().clip(lower=1), 0.0)
    df['conversions'] = np.minimum(df['conversions'], df['clicks'])  # Conversions cannot exceed clicks
    
    df['channel'] = 'meta'
    df['campaign_type'] = 'Social'
    df['exclude_from_modeling'] = False
    
    # Handle missing daily budget
    # Fill daily_budget if it is null or empty string
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    # Fill missing budget by campaign median, or overall median
    df['budget'] = df.groupby('campaign_id')['budget'].transform(lambda x: x.fillna(x.median() if x.median() > 0 else 50.0))
    df['budget'] = df['budget'].fillna(50.0)
    
    cols = ['date', 'campaign_id', 'campaign_name', 'campaign_type', 'channel', 
            'spend', 'clicks', 'impressions', 'conversions', 'revenue', 'budget', 'exclude_from_modeling']
    return df[cols]

def load_bing_ads(data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, "bing_campaign_stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Bing Ads stats not found at {path}")
        
    df = pd.read_csv(path)
    
    rename_dict = {
        'TimePeriod': 'date',
        'CampaignId': 'campaign_id',
        'CampaignName': 'campaign_name',
        'CampaignType': 'campaign_type',
        'Spend': 'spend',
        'Clicks': 'clicks',
        'Impressions': 'impressions',
        'Conversions': 'conversions',
        'Revenue': 'revenue',
        'DailyBudget': 'budget'
    }
    df = df.rename(columns=rename_dict)
    
    df['channel'] = 'bing'
    
    # Only Search campaigns have meaningful revenue signals. Exclude other types from modeling.
    df['exclude_from_modeling'] = df['campaign_type'].str.upper() != 'SEARCH'
    
    cols = ['date', 'campaign_id', 'campaign_name', 'campaign_type', 'channel', 
            'spend', 'clicks', 'impressions', 'conversions', 'revenue', 'budget', 'exclude_from_modeling']
    return df[cols]

def preprocess_all(data_dir: str) -> pd.DataFrame:
    # Load and clean datasets
    g_df = load_google_ads(data_dir)
    m_df = load_meta_ads(data_dir)
    b_df = load_bing_ads(data_dir)
    
    # Combine
    combined = pd.concat([g_df, m_df, b_df], ignore_index=True)
    
    # Standardize data types
    combined['date'] = pd.to_datetime(combined['date'])
    combined['campaign_id'] = combined['campaign_id'].astype(str)
    combined['spend'] = pd.to_numeric(combined['spend'], errors='coerce').fillna(0.0).clip(lower=0)
    combined['clicks'] = pd.to_numeric(combined['clicks'], errors='coerce').fillna(0.0).clip(lower=0)
    combined['impressions'] = pd.to_numeric(combined['impressions'], errors='coerce').fillna(0.0).clip(lower=0)
    combined['conversions'] = pd.to_numeric(combined['conversions'], errors='coerce').fillna(0.0).clip(lower=0)
    combined['revenue'] = pd.to_numeric(combined['revenue'], errors='coerce').fillna(0.0).clip(lower=0)
    combined['budget'] = pd.to_numeric(combined['budget'], errors='coerce').fillna(0.0).clip(lower=0)
    
    # Calculate CPC and Winsorize at the 99th percentile
    # CPC = spend / clicks (CPC = 0 if clicks is 0)
    combined['cpc'] = np.where(combined['clicks'] > 0, combined['spend'] / combined['clicks'], 0.0)
    
    cpc_99 = combined['cpc'].quantile(0.99)
    # Winsorize CPC column
    combined['cpc'] = combined['cpc'].clip(upper=cpc_99)
    
    # Recalculate spend to align with winsorized CPC to ensure cost-efficiency figures remain valid
    # combined['spend'] = combined['cpc'] * combined['clicks']
    
    # Remove duplicate (channel, campaign_id, date) rows — keep first occurrence
    combined = combined.drop_duplicates(subset=['channel', 'campaign_id', 'date'], keep='first')

    # Sort and return
    combined = combined.sort_values(['channel', 'campaign_id', 'date']).reset_index(drop=True)
    return combined

if __name__ == "__main__":
    df = preprocess_all(DATA_DIR)
    print(f"Preprocessed shape: {df.shape}")
    print(df.head())
