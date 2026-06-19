import os
import unittest
import pandas as pd
import numpy as np

# Add src to python path if needed (done implicitly by running from root)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import preprocess_all, load_google_ads, load_meta_ads, load_bing_ads
from aggregation import aggregate_to_weekly, aggregate_predictions_hierarchical
from features import create_features
from predict import predict_all_horizons, get_latest_state
from monte_carlo import run_portfolio_monte_carlo
from budget_simulator import fit_response_curves, run_budget_simulation
from llm_analyst import generate_analyst_report, monitor_drift, detect_anomalies

class TestForecastingPipeline(unittest.TestCase):
    
    def setUp(self):
        self.data_dir = "data"
        self.model_path = "pickle/model.pkl"
        
    def test_data_preprocessing(self):
        """Test that data loads and columns are standardized properly."""
        df = preprocess_all(self.data_dir)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        
        # Test required columns are present
        required_cols = ['date', 'campaign_id', 'campaign_name', 'campaign_type', 'channel', 
                         'spend', 'clicks', 'impressions', 'conversions', 'revenue', 'budget', 'exclude_from_modeling', 'cpc']
        for col in required_cols:
            self.assertIn(col, df.columns)
            
        # Test winsorization of CPC (should be capped close to 99th percentile)
        cpc_max = df['cpc'].max()
        cpc_99 = df['cpc'].quantile(0.99)
        self.assertLessEqual(cpc_max, cpc_99 + 1e-3)
        
        # Test channel-specific rules
        google = df[df['channel'] == 'google']
        display_google = google[google['campaign_type'].str.upper() == 'DISPLAY']
        if len(display_google) > 0:
            self.assertTrue(display_google['exclude_from_modeling'].all())
            
        bing = df[df['channel'] == 'bing']
        non_search_bing = bing[bing['campaign_type'].str.upper() != 'SEARCH']
        if len(non_search_bing) > 0:
            self.assertTrue(non_search_bing['exclude_from_modeling'].all())
            
    def test_weekly_aggregation(self):
        """Test daily to weekly aggregation."""
        daily = preprocess_all(self.data_dir)
        weekly = aggregate_to_weekly(daily)
        self.assertIsNotNone(weekly)
        self.assertLess(len(weekly), len(daily))
        
        # Check that dates are Monday-starts
        # Monday is weekday 0 in pandas
        self.assertTrue((weekly['date'].dt.weekday == 0).all())
        
    def test_feature_engineering(self):
        """Test feature generation, lags, rolling means, and categorical casting."""
        daily = preprocess_all(self.data_dir)
        weekly = aggregate_to_weekly(daily)
        feat_df = create_features(weekly, is_training=True)
        
        # Test lag columns
        self.assertIn('spend_lag_1', feat_df.columns)
        self.assertIn('revenue_lag_8', feat_df.columns)
        
        # Test rolling columns
        self.assertIn('spend_rolling_4w', feat_df.columns)
        self.assertIn('revenue_rolling_12w', feat_df.columns)
        
        # Test targets
        self.assertIn('target_revenue_30d', feat_df.columns)
        self.assertIn('target_revenue_90d', feat_df.columns)
        
        # Test categoricals
        self.assertEqual(feat_df['channel'].dtype, 'category')
        self.assertEqual(feat_df['campaign_type'].dtype, 'category')
        
    def test_prediction_monotoncity_and_reconciliation(self):
        """Test that predictions aggregate bottom-up and bounds are monotonic."""
        preds = predict_all_horizons(self.data_dir, self.model_path)
        self.assertIsNotNone(preds)
        
        # Check hierarchy levels
        levels = preds['level'].unique()
        self.assertIn('campaign', levels)
        self.assertIn('channel', levels)
        self.assertIn('total', levels)
        
        # Check monotonicity constraints
        self.assertTrue((preds['revenue_p10'] <= preds['revenue']).all())
        self.assertTrue((preds['revenue'] <= preds['revenue_p90']).all())
        self.assertTrue((preds['roas_p10'] <= preds['roas']).all())
        self.assertTrue((preds['roas'] <= preds['roas_p90']).all())
        
        # Verify exact mathematical reconciliation
        for h in ['30d', '60d', '90d']:
            h_sub = preds[preds['horizon'] == h]
            campaign_spend = h_sub[h_sub['level'] == 'campaign']['spend'].sum()
            total_spend = h_sub[h_sub['level'] == 'total']['spend'].values[0]
            self.assertAlmostEqual(campaign_spend, total_spend, places=2)
            
            campaign_rev = h_sub[h_sub['level'] == 'campaign']['revenue'].sum()
            total_rev = h_sub[h_sub['level'] == 'total']['revenue'].values[0]
            self.assertAlmostEqual(campaign_rev, total_rev, places=2)
            
    def test_monte_carlo(self):
        """Test that Monte Carlo runs and respects P10 <= P50 <= P90 bounds."""
        daily = preprocess_all(self.data_dir)
        weekly = aggregate_to_weekly(daily)
        mc_res = run_portfolio_monte_carlo(weekly, {'google': 0.10, 'meta': -0.05, 'bing': 0.0}, horizon_weeks=4)
        
        self.assertIn('total', mc_res)
        total_res = mc_res['total']
        self.assertIn('expected_revenue', total_res)
        self.assertTrue(total_res['revenue_p10'] <= total_res['revenue_p50'])
        self.assertTrue(total_res['revenue_p50'] <= total_res['revenue_p90'])
        
    def test_budget_simulation_and_response_curves(self):
        """Test budget scenario engine and curves."""
        sim_res = run_budget_simulation(self.data_dir, self.model_path, google_change=0.20, meta_change=-0.10, bing_change=0.0)
        self.assertIn('predictions', sim_res)
        self.assertIn('curves', sim_res)
        
        curves = sim_res['curves']
        self.assertIn('SEARCH', curves)
        self.assertTrue(curves['SEARCH']['max_observed_spend'] > 0)
        
    def test_llm_analyst_report(self):
        """Test LLM analyst report schema, data drift PSI, and anomalies checks."""
        drift = monitor_drift(self.data_dir)
        self.assertIn('spend', drift)
        self.assertIn('psi', drift['spend'])
        
        anom = detect_anomalies(self.data_dir, self.model_path)
        self.assertIsInstance(anom, list)
        
        # Test schema
        sim_res = run_budget_simulation(self.data_dir, self.model_path)
        report = generate_analyst_report(self.data_dir, self.model_path, sim_res, {}, {})
        
        required_schema = ['executive_summary', 'forecast_confidence', 'top_risks', 'opportunities', 
                           'budget_recommendations', 'channel_insights', 'risk_radar', 'anomaly_detection', 
                           'data_drift', 'confidence_assessment', 'key_assumptions', 'forecast_limitations']
        for k in required_schema:
            self.assertIn(k, report)

if __name__ == "__main__":
    unittest.main()
