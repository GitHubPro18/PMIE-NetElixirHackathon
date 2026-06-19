import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from preprocess import preprocess_all
from aggregation import aggregate_to_weekly
from features import create_features
from budget_simulator import run_budget_simulation
from shap_explainer import compute_shap_explanations
from config import DRIFT_THRESHOLDS
from logger import get_logger
from config import DATA_DIR, MODEL_PATH


logger = get_logger(__name__)

# Load environment variables from .env file
load_dotenv()

ANALYST_REPORT_FIELDS = {
    "executive_summary": str,
    "opportunities": list,
    "budget_recommendations": list,
    "channel_insights": list,
    "confidence_assessment": str,
    "key_assumptions": list,
    "forecast_limitations": list,
}


def _clean_json_response(text: str) -> str:
    """
    Extracts JSON from plain or fenced LLM responses.
    """
    if not text:
        raise ValueError("LLM response was empty")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 3:
            cleaned = parts[1]
            if cleaned.lstrip().lower().startswith("json"):
                cleaned = cleaned.lstrip()[4:]
    return cleaned.strip()


def _validate_llm_report(payload: dict) -> dict:
    """
    Keeps the memo schema predictable for Streamlit and tests.
    """
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")

    validated = {}
    for field, expected_type in ANALYST_REPORT_FIELDS.items():
        value = payload.get(field)
        if expected_type is str:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"LLM field `{field}` must be a non-empty string")
            validated[field] = value.strip()
        else:
            if not isinstance(value, list) or not value:
                raise ValueError(f"LLM field `{field}` must be a non-empty list")
            validated[field] = [str(item).strip() for item in value if str(item).strip()]
            if not validated[field]:
                raise ValueError(f"LLM field `{field}` must include non-empty items")
    return validated


def _safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _prediction_row(predictions: pd.DataFrame, horizon: str, level: str, **filters) -> dict:
    rows = predictions[(predictions["horizon"] == horizon) & (predictions["level"] == level)]
    for column, value in filters.items():
        rows = rows[rows[column] == value]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _forecast_context(forecast_results: dict, shap_results: dict, mc_results: dict) -> dict:
    predictions = forecast_results["predictions"]
    total_30d = _prediction_row(predictions, "30d", "total")
    channel_rows = predictions[
        (predictions["horizon"] == "30d") & (predictions["level"] == "channel")
    ].copy()

    channels = []
    for _, row in channel_rows.sort_values("revenue", ascending=False).iterrows():
        channels.append({
            "channel": row.get("channel"),
            "spend": _safe_float(row.get("spend")),
            "revenue": _safe_float(row.get("revenue")),
            "roas": _safe_float(row.get("roas")),
            "revenue_p10": _safe_float(row.get("revenue_p10")),
            "revenue_p90": _safe_float(row.get("revenue_p90")),
        })

    shap_summary = []
    if isinstance(shap_results, dict):
        shap_summary = shap_results.get("revenue", {}).get("drivers", [])

    mc_summary = {}
    if isinstance(mc_results, dict):
        mc_total = mc_results.get("total", {})
        mc_summary = {
            "expected_revenue": _safe_float(mc_total.get("expected_revenue")),
            "revenue_p10": _safe_float(mc_total.get("revenue_p10")),
            "revenue_p50": _safe_float(mc_total.get("revenue_p50")),
            "revenue_p90": _safe_float(mc_total.get("revenue_p90")),
            "expected_roas": _safe_float(mc_total.get("expected_roas")),
        }

    spend = _safe_float(total_30d.get("spend"))
    revenue = _safe_float(total_30d.get("revenue"))
    roas = _safe_float(total_30d.get("roas"), revenue / spend if spend > 0 else 0.0)

    return {
        "forecast_summary_30d": {
            "total_spend": spend,
            "total_revenue": revenue,
            "total_roas": roas,
            "confidence_score": _safe_float(total_30d.get("confidence_score")),
        },
        "channel_forecasts_30d": channels,
        "monte_carlo_30d": mc_summary,
        "shap_revenue_drivers": shap_summary,
    }


def _data_derived_report(context: dict, drift_metrics: dict, anomalies: list, risk_radar: list) -> dict:
    summary = context["forecast_summary_30d"]
    channels = context["channel_forecasts_30d"]
    top_channel = channels[0] if channels else {}
    weakest_channel = min(channels, key=lambda c: c.get("roas", 0.0), default={})
    top_driver = (context.get("shap_revenue_drivers") or [{}])[0]

    anomaly_text = (
        f"{len(anomalies)} anomaly point(s) were detected in the last 90 days"
        if anomalies else
        "No anomaly points were detected in the last 90 days"
    )
    drift_count = sum(1 for values in drift_metrics.values() if values.get("status") != "Stable")

    opportunities = []
    if top_channel:
        opportunities.append(
            f"Prioritize {str(top_channel.get('channel', 'the leading channel')).title()} because it has the largest 30-day revenue forecast (${top_channel.get('revenue', 0):,.2f}) at {top_channel.get('roas', 0):.2f} ROAS."
        )
    if top_driver:
        opportunities.append(
            f"Use the strongest revenue driver, {top_driver.get('feature', 'the top SHAP feature')}, to focus bid and budget decisions."
        )
    if weakest_channel:
        opportunities.append(
            f"Review {str(weakest_channel.get('channel', 'the weakest channel')).title()} efficiency because it has the lowest channel ROAS in the 30-day forecast ({weakest_channel.get('roas', 0):.2f})."
        )

    budget_recs = []
    for channel in channels:
        action = "protect or scale" if channel.get("roas", 0) >= summary.get("total_roas", 0) else "cap or test incrementally"
        budget_recs.append(
            f"{action.capitalize()} {str(channel.get('channel', 'channel')).title()} spend: forecast ROAS is {channel.get('roas', 0):.2f} versus portfolio ROAS {summary.get('total_roas', 0):.2f}."
        )

    channel_insights = [
        f"{str(c.get('channel', 'Channel')).title()} is forecast to produce ${c.get('revenue', 0):,.2f} revenue from ${c.get('spend', 0):,.2f} spend, with an 80% interval of ${c.get('revenue_p10', 0):,.2f}-${c.get('revenue_p90', 0):,.2f}."
        for c in channels
    ]

    return {
        "executive_summary": (
            f"The 30-day forecast projects ${summary['total_revenue']:,.2f} revenue on "
            f"${summary['total_spend']:,.2f} spend, for {summary['total_roas']:.2f} ROAS. "
            f"{anomaly_text}, and {drift_count} monitored metric(s) currently show drift."
        ),
        "opportunities": opportunities or ["No ranked opportunity was available from the current forecast data."],
        "budget_recommendations": budget_recs or ["No channel budget recommendation was available from the current forecast data."],
        "channel_insights": channel_insights or ["No channel-level forecast rows were available for the current scenario."],
        "confidence_assessment": (
            f"Forecast confidence is based on split-conformal confidence scores; the current mean score is "
            f"{summary.get('confidence_score', 0) * 100:.1f}%."
        ),
        "key_assumptions": [
            "The report uses the uploaded campaign logs, trained model artifact, scenario spend multipliers, conformal intervals, Monte Carlo output, and SHAP drivers available at runtime.",
            "Channel comparisons use the 30-day forecast horizon unless the UI section states otherwise.",
        ],
        "forecast_limitations": [
            "Recommendations are bounded by the historical data coverage and the trained model artifact loaded for this run.",
            "Scenario forecasts become less reliable when spend changes move far outside observed channel and campaign-type ranges.",
        ],
    }


def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_bins: int = 5) -> float:
    """
    Computes the Population Stability Index (PSI) between two distributions.
    """
    # Filter out NaNs
    baseline = baseline[~np.isnan(baseline)]
    target = target[~np.isnan(target)]
    
    if len(baseline) == 0 or len(target) == 0:
        return 0.0
        
    # Get bin edges based on baseline percentiles
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(baseline, percentiles)
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0
        
    # Calculate frequencies
    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    target_counts, _ = np.histogram(target, bins=bin_edges)
    
    # Convert to proportions with laplace smoothing (epsilon) to avoid log of 0
    eps = 1e-4
    b_props = (baseline_counts + eps) / (len(baseline) + eps * len(baseline_counts))
    t_props = (target_counts + eps) / (len(target) + eps * len(target_counts))
    
    # Calculate PSI
    psi_value = np.sum((t_props - b_props) * np.log(t_props / b_props))
    return float(psi_value)

def detect_anomalies(data_dir: str, model_path: str) -> list:
    """
    Detects anomalies over the last 90 days by checking if actual revenue 
    falls outside conformal prediction intervals.
    """
    import pickle
    from predict import predict_all_horizons
    
    # Load model conformal quantiles
    if not os.path.exists(model_path):
        return []
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    q_rel = model_dict.get('conformal_quantiles', {}).get('30d', 0.25)
    
    # Get historical data
    daily = preprocess_all(data_dir)
    weekly = aggregate_to_weekly(daily)
    feat_df = create_features(weekly, is_training=True)
    
    # Group to total level for aggregate anomaly detection
    total_weekly = feat_df.groupby('date').agg({
        'spend': 'sum',
        'revenue': 'sum'
    }).reset_index().sort_values('date')
    
    # Take last 90 days (approx 13 weeks)
    last_13 = total_weekly.tail(13).copy()
    if len(last_13) == 0:
        return []
        
    # Use a simple rolling forecast standard error or the conformal quantile to build historical predicted bounds
    # Since we don't have walk-forward model predictions stored, we can train a quick validation model 
    # or use a median-based baseline predictor to establish expected revenue bounds.
    # A simple baseline: expected revenue is rolling 4-week mean of revenue
    last_13['expected_revenue'] = last_13['revenue'].rolling(window=4, min_periods=1).mean()
    last_13['p10'] = last_13['expected_revenue'] * (1.0 - q_rel)
    last_13['p90'] = last_13['expected_revenue'] * (1.0 + q_rel)
    
    anomalies = []
    for _, row in last_13.iterrows():
        act = row['revenue']
        p10 = row['p10']
        p90 = row['p90']
        if act < p10 or act > p90:
            deviation = (act - row['expected_revenue']) / (row['expected_revenue'] + 1.0)
            anomalies.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'actual': float(act),
                'expected': float(row['expected_revenue']),
                'p10': float(p10),
                'p90': float(p90),
                'deviation_pct': float(deviation * 100)
            })
    return anomalies

def monitor_drift(data_dir: str) -> dict:
    """
    Computes PSI for spend, CTR, CVR, and ROAS to identify data drift.
    """
    daily = preprocess_all(data_dir)
    weekly = aggregate_to_weekly(daily)
    
    weekly['roas'] = np.where(weekly['spend'] > 0, weekly['revenue'] / weekly['spend'], 0.0)
    weekly['ctr'] = np.where(weekly['impressions'] > 0, weekly['clicks'] / weekly['impressions'], 0.0)
    weekly['cvr'] = np.where(weekly['clicks'] > 0, weekly['conversions'] / weekly['clicks'], 0.0)
    
    # Chronological split: first 70% as baseline, last 30% as target
    weekly = weekly.sort_values('date').reset_index(drop=True)
    split_idx = int(len(weekly) * 0.7)
    
    baseline_df = weekly.iloc[:split_idx]
    target_df = weekly.iloc[split_idx:]
    
    metrics = ['spend', 'ctr', 'cvr', 'roas']
    drift_results = {}
    
    for m in metrics:
        b_vals = baseline_df[m].values
        t_vals = target_df[m].values
        psi = calculate_psi(b_vals, t_vals)
        
        status = "Stable"
        if psi > DRIFT_THRESHOLDS.get("moderate", 0.25):
            status = "High Drift"
        elif psi > DRIFT_THRESHOLDS.get("stable", 0.10):
            status = "Moderate Drift"
            
        drift_results[m] = {
            'psi': psi,
            'status': status
        }
    return drift_results

def generate_analyst_report(data_dir: str, model_path: str, forecast_results: dict, shap_results: dict, mc_results: dict) -> dict:
    """
    Generates a structured marketing analyst report including risk radar, 
    anomalies, and drift metrics. Uses Gemini for the memo when configured,
    otherwise returns an explicitly marked data-derived fallback.
    """
    drift_metrics = monitor_drift(data_dir)
    anomalies = detect_anomalies(data_dir, model_path)
    context = _forecast_context(forecast_results, shap_results, mc_results)
    total_roas_30d = context["forecast_summary_30d"]["total_roas"]

    risk_radar = []

    for metric, res in drift_metrics.items():
        if res['status'] == "High Drift":
            risk_radar.append({
                'risk': f"Data Drift in {metric.upper()}",
                'severity': "High",
                'mitigation': f"Retrain forecasting models immediately on recent data. The historical distribution of {metric} has shifted significantly (PSI = {res['psi']:.3f})."
            })
        elif res['status'] == "Moderate Drift":
            risk_radar.append({
                'risk': f"Moderate Shift in {metric.upper()}",
                'severity': "Medium",
                'mitigation': f"Monitor campaign structure for {metric}. Stable but showing initial indicators of divergence (PSI = {res['psi']:.3f})."
            })

    if len(anomalies) > 0:
        latest_anomaly = anomalies[-1]
        dev_word = "undershot" if latest_anomaly['deviation_pct'] < 0 else "overshot"
        risk_radar.append({
            'risk': f"Revenue Anomaly Detected ({latest_anomaly['date']})",
            'severity': "Medium" if abs(latest_anomaly['deviation_pct']) < 25 else "High",
            'mitigation': f"Investigate channel tracking on {latest_anomaly['date']}. Actual revenue {dev_word} the forecast by {abs(latest_anomaly['deviation_pct']):.1f}%."
        })

    if total_roas_30d < 1.5:
        risk_radar.append({
            'risk': "Low ROAS Performance",
            'severity': "High",
            'mitigation': "Prioritize channels and campaign types with above-portfolio ROAS before adding incremental spend."
        })
    else:
        risk_radar.append({
            'risk': "Ad-spend Saturation",
            'severity': "Low",
            'mitigation': "Watch for widening forecast intervals as scenario spend exceeds historical channel ranges."
        })

    memo = _data_derived_report(context, drift_metrics, anomalies, risk_radar)
    llm_status = "not_configured"
    llm_error = None
    api_key = os.environ.get("GEMINI_API_KEY")

    if api_key:
        prompt = f"""
You are an expert performance marketing analyst.

Use only the JSON data below. Do not invent channels, budgets, platform names, anomalies, or recommendations that are not supported by the data.

Forecast and model context:
{json.dumps(context, indent=2)}

Recent anomalies:
{json.dumps(anomalies, indent=2)}

Data drift metrics:
{json.dumps(drift_metrics, indent=2)}

Risk radar:
{json.dumps(risk_radar, indent=2)}

Return only a valid JSON object with this exact schema:
{{
  "executive_summary": "short paragraph grounded in the forecast numbers",
  "opportunities": ["data-backed opportunity", "data-backed opportunity"],
  "budget_recommendations": ["data-backed recommendation", "data-backed recommendation"],
  "channel_insights": ["channel-specific insight", "channel-specific insight"],
  "confidence_assessment": "short explanation of confidence using intervals, drift, anomalies, and Monte Carlo where present",
  "key_assumptions": ["assumption directly implied by the supplied inputs"],
  "forecast_limitations": ["limitation directly implied by the supplied inputs"]
}}
"""

        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )

            raw_text = _clean_json_response(response.text)
            memo = _validate_llm_report(json.loads(raw_text))
            llm_status = "generated"
        except Exception as e:
            llm_status = "fallback"
            llm_error = str(e)
            logger.error(f"Failed to get LLM interpretation: {e}")

    report = {
        'executive_summary': memo['executive_summary'],
        'forecast_confidence': f"{forecast_results['predictions']['confidence_score'].mean() * 100:.1f}%",
        'top_risks': [r['risk'] for r in risk_radar if r['severity'] in ['High', 'Medium']],
        'opportunities': memo['opportunities'],
        'budget_recommendations': memo['budget_recommendations'],
        'channel_insights': memo['channel_insights'],
        'risk_radar': risk_radar,
        'anomaly_detection': {
            'recent_anomalies': anomalies,
            'status': "Stable" if len(anomalies) == 0 else f"{len(anomalies)} Anomalies Detected"
        },
        'data_drift': drift_metrics,
        'confidence_assessment': memo['confidence_assessment'],
        'key_assumptions': memo['key_assumptions'],
        'forecast_limitations': memo['forecast_limitations'],
        'llm_status': llm_status,
        'llm_error': llm_error,
    }
    
    return report

if __name__ == "__main__":
    
    print("Running forecast for LLM analyst test...")
    sim_res = run_budget_simulation(DATA_DIR, MODEL_PATH)
    shap_str = compute_shap_explanations(DATA_DIR, MODEL_PATH)
    shap_dict = json.loads(shap_str)
    
    report = generate_analyst_report(DATA_DIR, MODEL_PATH, sim_res, shap_dict, {})
    print(json.dumps(report, indent=2))
