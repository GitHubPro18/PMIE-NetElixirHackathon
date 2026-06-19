import os
import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from preprocess import preprocess_all
from aggregation import aggregate_to_weekly
from features import create_features
from config import DATA_DIR, MODEL_PATH


def get_feature_lists(horizon: str):
    """
    Returns base features list and categorical features.
    """
    features_base = [
        'week_of_year', 'month', 'quarter',
        'campaign_id', 'campaign_type', 'channel'
    ]
    lag_cols = ['spend', 'revenue', 'clicks', 'impressions', 'conversions', 'roas']
    for col in lag_cols:
        for lag in [1, 2, 4, 8]:
            features_base.append(f'{col}_lag_{lag}')
        for w in [4, 8, 12]:
            features_base.append(f'{col}_rolling_{w}w')
            
    # Include the cumulative spend for the forecasting horizon
    features_all = features_base + [f'horizon_spend_{horizon}']
    categorical_features = ['campaign_id', 'campaign_type', 'channel']
    
    return features_all, categorical_features

def train_horizon_model(df: pd.DataFrame, horizon: str):
    """
    Trains a LightGBM regressor for a specific horizon (30d, 60d, or 90d).
    """
    target_col = f'target_revenue_{horizon}'
    features_all, categorical_features = get_feature_lists(horizon)
    
    # Filter for modeling and drop missing targets/features
    model_df = df[~df['exclude_from_modeling']].copy()
    model_df = model_df.dropna(subset=[target_col] + features_all)
    
    if len(model_df) == 0:
        raise ValueError(f"No training samples left for horizon {horizon}")
        
    # Chronological sort
    model_df = model_df.sort_values('date').reset_index(drop=True)
    
    X = model_df[features_all]
    y = model_df[target_col]
    
    print("*" * 60)
    print(f"Training Revenue model for horizon: {horizon}")
    print(f"Features count: {X.shape[1]}, Training rows: {X.shape[0]}")
    
    # TimeSeriesSplit Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Build dataset
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_features)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=categorical_features)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'verbosity': -1,
            'random_state': 42 + fold
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=300,
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        preds = model.predict(X_val)
        # Business Constraint: predictions >= 0
        preds = np.clip(preds, 0, None)
        
        mae = mean_absolute_error(y_val, preds)
        rmse = root_mean_squared_error(y_val, preds)
        r2 = r2_score(y_val, preds)
        
        cv_scores.append((mae, rmse, r2))
        print(f"  Fold {fold+1} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.3f}")
        
    avg_mae = np.mean([s[0] for s in cv_scores])
    avg_rmse = np.mean([s[1] for s in cv_scores])
    avg_r2 = np.mean([s[2] for s in cv_scores])
    print(f"Average CV Scores - MAE: {avg_mae:.2f}, RMSE: {avg_rmse:.2f}, R2: {avg_r2:.3f}")
    
    # Train final model on entire dataset
    full_train_data = lgb.Dataset(X, label=y, categorical_feature=categorical_features)
    final_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'verbosity': -1,
        'random_state': 100
    }
    
    # Use average boost rounds from CV (or simple fixed amount)
    final_model = lgb.train(
        final_params,
        full_train_data,
        num_boost_round=150
    )
    
    return final_model, features_all

def main():
    print("Preprocessing data...")
    daily = preprocess_all(DATA_DIR)
    weekly = aggregate_to_weekly(daily)
    
    print("Generating features...")
    feat_df = create_features(weekly, is_training=True)
    
    models = {}
    feature_cols = {}
    
    for h in ['30d', '60d', '90d']:
        model, f_cols = train_horizon_model(feat_df, h)
        models[h] = model
        feature_cols[h] = f_cols
        
    # Save model dict to pickle
    os.makedirs('pickle', exist_ok=True)
    pickle_path = MODEL_PATH
    
    # Save whatever exists (if there is an existing pickle with roas, load and update)
    existing_pkl = {}
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, 'rb') as f:
                existing_pkl = pickle.load(f)
        except Exception:
            pass
            
    existing_pkl.update({
        'revenue_30d': models['30d'],
        'revenue_60d': models['60d'],
        'revenue_90d': models['90d'],
        'feature_cols_revenue': feature_cols
    })
    
    with open(pickle_path, 'wb') as f:
        pickle.dump(existing_pkl, f)
    print(f"Revenue models successfully saved to {pickle_path}!")

if __name__ == "__main__":
    main()
