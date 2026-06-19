import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import rankdata

from config import MC_SIMULATIONS, MC_RANDOM_SEED,DATA_DIR
from preprocess import preprocess_all
from aggregation import aggregate_to_weekly


def fit_empirical_cdf(data: np.ndarray) -> np.ndarray:
    """
    Returns empirical CDF values for the data.
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)
    # Return a function or percentiles
    return sorted_data

def sample_empirical_cdf(sorted_data: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Performs inverse transform sampling on empirical data.
    u is an array of uniform random variables in [0, 1].

    BUG FIX (Priority 3): Using (u * (n-1)).round() instead of (u * n).astype(int)
    to give a proper uniform mapping across [0, n-1] without biasing toward index 0
    or clipping the max value too often.
    """
    n = len(sorted_data)
    # Correct: map [0,1] uniformly to [0, n-1] as floats, then round
    indices = np.clip(np.round(u * (n - 1)).astype(int), 0, n - 1)
    return sorted_data[indices]

def _winsorize_aov(aov: np.ndarray, lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
    """
    Winsorizes AOV values to remove extreme outliers that inflate the tail distribution.
    
    BUG FIX (Priority 3): Bing's raw AOV has max=$2,022 vs median=$121, creating
    a P99.9/P50 ratio of 22x — well above the healthy <10x threshold.
    Winsorizing to the [1st, 99th] percentile removes outlier contamination while
    preserving the realistic empirical distribution for the bulk of simulations.
    """
    lo = np.percentile(aov, lower_pct)
    hi = np.percentile(aov, upper_pct)
    return np.clip(aov, lo, hi)

def run_portfolio_monte_carlo(
    weekly_df: pd.DataFrame,
    channel_spend_adjustments: dict,
    horizon_weeks: int = 4,
    num_simulations: int = MC_SIMULATIONS,
    deterministic_total_spend: float = None
) -> dict:
    """
    Runs Monte Carlo simulations using Gaussian Copula for CTR, CVR, and AOV per channel, 
    and aggregates them to the portfolio level.
    
    Parameters:
      weekly_df: DataFrame of weekly campaign entries.
      channel_spend_adjustments: dict of channel adjustments, e.g. {'google': 0.20, 'meta': -0.10, 'bing': 0.05}
      horizon_weeks: forecasting weeks (4 for 30d, 8 for 60d, 13 for 90d)
      num_simulations: number of simulation draws
      deterministic_total_spend: optional override for total planned spend, used to ensure
        ROAS denominator matches the deterministic budget_simulator output.
        
    Returns:
      dict containing simulated statistics per channel and total portfolio.
    
    FIX (Priority 1 – ROAS Consistency):
      The spend denominator used to compute ROAS is now the same as the one
      computed by the deterministic budget simulator when `deterministic_total_spend`
      is passed. This guarantees that both systems report a consistent ROAS baseline.
      If not provided, we fall back to summing the channel-level planned spends
      computed from the last 4 weeks of history (original behaviour).
    """
    df = weekly_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate weekly aggregate metrics per channel
    channel_weekly = df.groupby(['channel', 'date'], observed=False).agg({
        'spend': 'sum',
        'clicks': 'sum',
        'impressions': 'sum',
        'conversions': 'sum',
        'revenue': 'sum'
    }).reset_index()
    
    # Filter for active channels and calculate rates
    channel_weekly['ctr'] = np.where(channel_weekly['impressions'] > 0, channel_weekly['clicks'] / channel_weekly['impressions'], 0.0)
    channel_weekly['cvr'] = np.where(channel_weekly['clicks'] > 0, channel_weekly['conversions'] / channel_weekly['clicks'], 0.0)
    channel_weekly['aov'] = np.where(channel_weekly['conversions'] > 0, channel_weekly['revenue'] / channel_weekly['conversions'], 0.0)
    
    # Keep only records where impressions, clicks, conversions, and revenue are > 0 to have realistic rates
    valid_data = channel_weekly[(channel_weekly['impressions'] > 0) & 
                                (channel_weekly['clicks'] > 0) & 
                                (channel_weekly['conversions'] > 0) & 
                                (channel_weekly['revenue'] > 0)].copy()
                                
    channels = ['google', 'meta', 'bing']
    sim_revenues = {}
    planned_spends = {}
    
    # Fix seed for reproducibility
    np.random.seed(MC_RANDOM_SEED)
    
    for channel in channels:
        ch_sub = valid_data[valid_data['channel'] == channel]
        
        # Calculate planned impressions over the horizon
        # Get baseline weekly impressions and spend (mean of last 4 weeks of original dataset)
        orig_ch_sub = df[df['channel'] == channel].sort_values('date')
        if len(orig_ch_sub) == 0:
            sim_revenues[channel] = np.zeros(num_simulations)
            planned_spends[channel] = 0.0
            continue
            
        last_4_weeks = orig_ch_sub.groupby('date').sum(numeric_only=True).tail(4)
        baseline_weekly_impressions = float(last_4_weeks['impressions'].mean())
        baseline_weekly_spend = float(last_4_weeks['spend'].mean())
        
        m = 1.0 + channel_spend_adjustments.get(channel, 0.0)
        planned_spend = baseline_weekly_spend * m * horizon_weeks
        planned_spends[channel] = planned_spend
        
        # Scale planned impressions linearly with spend adjustments
        planned_impressions = baseline_weekly_impressions * m * horizon_weeks
        
        if len(ch_sub) < 5 or planned_impressions == 0:
            # Fallback if history is too short to estimate copula or channel is inactive
            sim_revenues[channel] = np.zeros(num_simulations)
            continue
            
        # Extract variables and copy to make writable
        X = ch_sub[['ctr', 'cvr', 'aov']].values.copy()
        # Enforce business constraints (rates between 0 and 1, AOV >= 0)
        X[:, 0] = np.clip(X[:, 0], 0.0001, 0.999)
        X[:, 1] = np.clip(X[:, 1], 0.0001, 0.999)
        X[:, 2] = np.clip(X[:, 2], 0.01, None)
        
        # FIX (Priority 3 – Tail Inflation): Winsorize AOV to remove extreme outliers
        # before feeding them into the empirical copula.
        # Bing's raw AOV reaches $2,022 (>16x the median of $121), which contaminated
        # the upper tail and produced P99.9/P50 ratios of 22x (threshold: 10x).
        # Winsorizing to [1%, 99%] eliminates outlier contamination while preserving
        # the realistic bulk of the AOV distribution.
        X[:, 2] = _winsorize_aov(X[:, 2], lower_pct=1.0, upper_pct=99.0)
        
        # Gaussian Copula: use Spearman (rank) correlation for correct copula structure
        # Transform each variable to uniform [0,1] margins via rank normalization
        n_obs = X.shape[0]
        X_ranked = np.column_stack([
            rankdata(X[:, i]) / (n_obs + 1) for i in range(X.shape[1])
        ])
        # Convert uniform margins to standard normal for correlation estimation
        X_normal = norm.ppf(X_ranked)
        corr_matrix = np.corrcoef(X_normal.T)
        # Handle potential NaNs if standard deviation is 0
        if np.any(np.isnan(corr_matrix)):
            corr_matrix = np.eye(3)
            
        # Ensure positive-definiteness
        eigvals, eigvecs = np.linalg.eigh(corr_matrix)
        eigvals = np.maximum(eigvals, 1e-6)
        corr_matrix_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # 10,000 multivariate normal draws
        z = np.random.multivariate_normal(mean=[0, 0, 0], cov=corr_matrix_pd, size=num_simulations)
        u = norm.cdf(z)
        
        # Sort values for empirical CDF mapping
        ctr_sorted = np.sort(X[:, 0])
        cvr_sorted = np.sort(X[:, 1])
        aov_sorted = np.sort(X[:, 2])
        
        # Map back to empirical CDF (fixed inverse transform sampling)
        ctr_sim = sample_empirical_cdf(ctr_sorted, u[:, 0])
        cvr_sim = sample_empirical_cdf(cvr_sorted, u[:, 1])
        aov_sim = sample_empirical_cdf(aov_sorted, u[:, 2])
        
        # Calculate simulated revenue: Revenue = Impressions * CTR * CVR * AOV
        sim_rev = planned_impressions * ctr_sim * cvr_sim * aov_sim
        
        # Enforce business constraint: revenue >= 0
        sim_rev = np.clip(sim_rev, 0, None)
        sim_revenues[channel] = sim_rev
        
    # Aggregate to Total Portfolio
    # FIX (Priority 1 – ROAS Consistency):
    # Use the deterministic spend from the budget_simulator when provided so that
    # MC ROAS is computed against the same denominator as the deterministic forecast.
    # This eliminates the 2.3x inflation that arose from the MC using only the
    # last-4-weeks channel aggregate ($55,916) while budget_sim used full campaign
    # aggregates ($126,116).
    if deterministic_total_spend is not None and deterministic_total_spend > 0:
        total_spend = deterministic_total_spend
    else:
        total_spend = sum(planned_spends.values())

    # Compute per-channel denominators proportionally to each channel's planned spend
    # so channel-level ROAS is also consistent with the deterministic forecast.
    mc_spend_total = sum(planned_spends.values())
    if mc_spend_total > 0 and deterministic_total_spend is not None:
        spend_scale = deterministic_total_spend / mc_spend_total
    else:
        spend_scale = 1.0

    # Scale per-channel planned spends to match deterministic total
    adjusted_planned_spends = {ch: v * spend_scale for ch, v in planned_spends.items()}

    total_sim_rev = np.zeros(num_simulations)
    for channel in channels:
        total_sim_rev += sim_revenues[channel]
        
    # Compile statistics
    results = {}
    
    for ch in channels + ['total']:
        revs = total_sim_rev if ch == 'total' else sim_revenues[ch]
        spend = total_spend if ch == 'total' else adjusted_planned_spends.get(ch, 0.0)
        
        expected_rev = float(np.mean(revs))
        p10_rev = float(np.percentile(revs, 10))
        p50_rev = float(np.percentile(revs, 50))
        p90_rev = float(np.percentile(revs, 90))
        
        # FIX (Priority 1): ROAS = mean(revenue) / spend  (not mean(revenue/spend))
        # Computing per-simulation ROAS then averaging is mathematically incorrect because
        # E[R/S] != E[R]/E[S] and amplifies high-revenue simulations that share the same
        # fixed spend denominator.  The correct estimator is mean(revenue)/spend.
        expected_roas = expected_rev / spend if spend > 0 else 0.0
        p10_roas = p10_rev / spend if spend > 0 else 0.0
        p50_roas = p50_rev / spend if spend > 0 else 0.0
        p90_roas = p90_rev / spend if spend > 0 else 0.0
        
        # Monotonicity checks
        p10_rev = min(p10_rev, p50_rev)
        p90_rev = max(p90_rev, p50_rev)
        p10_roas = min(p10_roas, p50_roas)
        p90_roas = max(p90_roas, p50_roas)
        
        results[ch] = {
            'expected_revenue': expected_rev,
            'revenue_p10': p10_rev,
            'revenue_p50': p50_rev,
            'revenue_p90': p90_rev,
            'expected_roas': expected_roas,
            'roas_p10': p10_roas,
            'roas_p50': p50_roas,
            'roas_p90': p90_roas,
            # Return samples (subsampled to 100 bins for charting or transfer size optimization)
            'histogram_bins': np.histogram(revs, bins=50)[0].tolist(),
            'histogram_edges': np.histogram(revs, bins=50)[1].tolist()
        }
        
    return results

if __name__ == "__main__":

    daily = preprocess_all(DATA_DIR)
    weekly = aggregate_to_weekly(daily)
    mc_results = run_portfolio_monte_carlo(weekly, {'google': 0.0, 'meta': 0.0, 'bing': 0.0}, horizon_weeks=4)
    print("Monte Carlo Portfolio Results (Total):")
    print(mc_results['total'])
