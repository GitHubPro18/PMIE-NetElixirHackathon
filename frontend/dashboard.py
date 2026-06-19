import os
import sys
import json
import math
import inspect
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit.components.v1 as components

# Add src folder to python path for direct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from preprocess import preprocess_all, load_google_ads, load_meta_ads, load_bing_ads
from aggregation import aggregate_to_weekly
from predict import predict_all_horizons
from budget_simulator import run_budget_simulation, fit_response_curves, hill_function
from shap_explainer import compute_shap_explanations
from monte_carlo import run_portfolio_monte_carlo
from llm_analyst import generate_analyst_report, monitor_drift, detect_anomalies

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PMIE – Probabilistic Marketing Intelligence Engine",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⚡",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS  —  Clean SaaS light theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & Base ─────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
  color: #111827 !important;
}

/* ── App shell ────────────────────────────────────── */
.stApp {
  background: #F8FAFC !important;
}
.block-container {
  padding-top: 60px !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
  padding-bottom: 2rem !important;
  max-width: 100% !important;
}
.stAppHeader {
  background-color: transparent !important;
}
.stApp > header { background-color: transparent !important; }

/* ── Sidebar ──────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: #0F172A !important;
  border-right: 1px solid #1E293B !important;
}
section[data-testid="stSidebar"] * {
  color: #CBD5E1 !important;
}
section[data-testid="stSidebar"] .stRadio label {
  color: #94A3B8 !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 6px 10px !important;
  border-radius: 6px !important;
  transition: background 0.15s, color 0.15s !important;
  cursor: pointer !important;
}
section[data-testid="stSidebar"] .stRadio label:hover {
  background: #1E293B !important;
  color: #F1F5F9 !important;
}
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
section[data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"]:has(> p) {
  /* To keep text styled inside */
}
section[data-testid="stSidebar"] hr {
  border-color: #1E293B !important;
  margin: 12px 0 !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
  font-size: 11px !important;
  color: #475569 !important;
}

/* ── Top bar ──────────────────────────────────────── */
.top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 0 14px 0;
  border-bottom: 1px solid #E5E7EB;
  margin-bottom: 20px;
}
.top-bar-left { display: flex; align-items: baseline; gap: 10px; }
.top-bar-logo {
  font-size: 20px; font-weight: 700; color: #111827;
  letter-spacing: -0.5px;
}
.top-bar-sub {
  font-size: 13px; color: #6B7280; font-weight: 400;
}
.top-bar-pills { display: flex; gap: 6px; flex-wrap: wrap; }
.pill {
  font-size: 11px; font-weight: 500; color: #4F46E5;
  background: #EEF2FF; border: 1px solid #C7D2FE;
  border-radius: 4px; padding: 2px 8px;
  letter-spacing: 0;
}
.pill-green { color: #059669; background: #ECFDF5; border-color: #A7F3D0; }
.pill-gray  { color: #374151; background: #F3F4F6; border-color: #D1D5DB; }

/* ── Section headers ──────────────────────────────── */
.section-title {
  font-size: 15px; font-weight: 600; color: #111827;
  margin: 20px 0 12px 0;
}
.section-sub {
  font-size: 13px; color: #6B7280; margin-top: -8px; margin-bottom: 12px;
}

/* ── Cards ────────────────────────────────────────── */
.card {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,.07);
}
.card-sm {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 16px 18px;
  box-shadow: 0 1px 3px rgba(0,0,0,.07);
}

/* ── KPI Cards ────────────────────────────────────── */
.kpi-card {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 18px 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
  height: 100%;
}
.kpi-label {
  font-size: 12px; font-weight: 500; color: #6B7280;
  margin-bottom: 6px; text-transform: none; letter-spacing: 0;
}
.kpi-value {
  font-size: 28px; font-weight: 700; color: #111827;
  line-height: 1.1; letter-spacing: -0.5px;
}
.kpi-value-primary { color: #4F46E5 !important; }
.kpi-value-success { color: #059669 !important; }
.kpi-value-warning { color: #D97706 !important; }
.kpi-value-danger  { color: #DC2626 !important; }
.kpi-delta {
  font-size: 12px; font-weight: 500; margin-top: 5px;
  display: inline-flex; align-items: center; gap: 3px;
}
.kpi-delta-up   { color: #059669; }
.kpi-delta-down { color: #DC2626; }
.kpi-delta-flat { color: #6B7280; }

/* ── Executive brief ──────────────────────────────── */
.exec-brief {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 24px 28px;
  box-shadow: 0 1px 3px rgba(0,0,0,.07);
  margin-bottom: 20px;
}
.exec-headline {
  font-size: 17px; font-weight: 600; color: #111827;
  margin-bottom: 10px; line-height: 1.4;
}
.exec-body {
  font-size: 14px; color: #374151; line-height: 1.7; margin-bottom: 16px;
}
.exec-chips { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
.exec-chip {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 12px; border-radius: 20px;
  font-size: 12px; font-weight: 500;
}
.chip-indigo { background: #EEF2FF; color: #4F46E5; border: 1px solid #C7D2FE; }
.chip-green  { background: #ECFDF5; color: #059669; border: 1px solid #A7F3D0; }
.chip-amber  { background: #FFFBEB; color: #D97706; border: 1px solid #FDE68A; }
.chip-red    { background: #FEF2F2; color: #DC2626; border: 1px solid #FECACA; }
.chip-blue   { background: #EFF6FF; color: #2563EB; border: 1px solid #BFDBFE; }
.exec-col-title {
  font-size: 11px; font-weight: 600; color: #9CA3AF;
  text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px;
}
.exec-item {
  font-size: 13px; color: #374151; padding: 5px 0;
  border-bottom: 1px solid #F3F4F6; line-height: 1.5;
}
.exec-item:last-child { border-bottom: none; }

/* ── Reliability tiles ────────────────────────────── */
.rel-tile {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 16px;
  padding: 20px 24px;
  height: 140px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  box-shadow: 0 1px 3px rgba(0,0,0,.05);
  margin-bottom: 16px;
}
.rel-tile-title { font-size: 13px; font-weight: 500; color: #6B7280; margin-bottom: 8px; }
.rel-tile-value { font-size: 24px; font-weight: 700; color: #111827; margin-bottom: 4px; }
.rel-tile-sub   { font-size: 12px; color: #9CA3AF; }
.status-dot {
  display: inline-block; width: 7px; height: 7px;
  border-radius: 50%; margin-right: 5px;
}
.dot-green  { background: #10B981; }
.dot-amber  { background: #F59E0B; }
.dot-red    { background: #EF4444; }

/* ── Monte Carlo cards ────────────────────────────── */
.mc-card {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 16px 14px;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
  text-align: center;
}
.mc-label { font-size: 11px; font-weight: 500; color: #9CA3AF; margin-bottom: 6px; }
.mc-value { font-size: 22px; font-weight: 700; }
.mc-sub   { font-size: 10px; color: #9CA3AF; margin-top: 4px; line-height: 1.4; }
.mc-worst  { color: #DC2626; }
.mc-likely { color: #D97706; }
.mc-best   { color: #059669; }
.mc-exp    { color: #4F46E5; }
.mc-roas   { color: #7C3AED; }

/* ── SHAP driver cards ────────────────────────────── */
.driver-card {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-left: 3px solid;
  border-radius: 8px;
  padding: 14px 16px;
  margin-bottom: 10px;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.driver-card-spend { border-left-color: #4F46E5; }
.driver-card-eff   { border-left-color: #7C3AED; }
.driver-card-seas  { border-left-color: #D97706; }
.driver-title { font-size: 13px; font-weight: 600; color: #111827; margin-bottom: 3px; }
.driver-feat  { font-size: 11px; color: #6B7280; margin-bottom: 5px; }
.driver-desc  { font-size: 12px; color: #374151; line-height: 1.5; }
.driver-badge {
  display: inline-block; font-size: 10px; font-weight: 600;
  padding: 1px 7px; border-radius: 3px; margin-left: 6px;
}
.badge-spend { background: #EEF2FF; color: #4F46E5; }
.badge-eff   { background: #F5F3FF; color: #7C3AED; }
.badge-seas  { background: #FFFBEB; color: #D97706; }

/* ── AI Memo sections ─────────────────────────────── */
.memo-section {
  background: #FFFFFF;
  border: 1px solid #E5E7EB;
  border-radius: 8px;
  padding: 18px 22px;
  margin-bottom: 10px;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.memo-title {
  font-size: 12px; font-weight: 600; color: #374151;
  text-transform: uppercase; letter-spacing: 0.7px;
  margin-bottom: 12px; display: flex; align-items: center; gap: 6px;
}
.memo-icon {
  width: 20px; height: 20px; border-radius: 4px;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 11px;
}
.mi-blue   { background: #EFF6FF; }
.mi-green  { background: #ECFDF5; }
.mi-indigo { background: #EEF2FF; }
.mi-red    { background: #FEF2F2; }
.mi-amber  { background: #FFFBEB; }
.memo-body { font-size: 13px; color: #374151; line-height: 1.7; }
.memo-item {
  display: flex; align-items: flex-start; gap: 8px;
  padding: 6px 0; border-bottom: 1px solid #F9FAFB;
  font-size: 13px; color: #374151;
}
.memo-item:last-child { border-bottom: none; }
.memo-risk-badge {
  display: inline-block; padding: 1px 7px; border-radius: 3px;
  font-size: 10px; font-weight: 600; text-transform: uppercase; margin-right: 5px;
}
.mbr-high   { background: #FEE2E2; color: #DC2626; }
.mbr-medium { background: #FEF3C7; color: #D97706; }
.mbr-low    { background: #D1FAE5; color: #059669; }

/* ── Severity badges ──────────────────────────────── */
.sev-badge {
  display: inline-block; padding: 1px 8px; border-radius: 3px;
  font-size: 10px; font-weight: 600; text-transform: uppercase;
}
.sev-critical { background: #FEE2E2; color: #DC2626; }
.sev-high     { background: #FEE2E2; color: #B91C1C; }
.sev-medium   { background: #FEF3C7; color: #D97706; }
.sev-low      { background: #D1FAE5; color: #059669; }
.psi-stable   { background: #D1FAE5; color: #059669; padding: 1px 8px;
  border-radius: 3px; font-size: 10px; font-weight: 600; display: inline-block; }
.psi-moderate { background: #FEF3C7; color: #D97706; padding: 1px 8px;
  border-radius: 3px; font-size: 10px; font-weight: 600; display: inline-block; }
.psi-high-drift { background: #FEE2E2; color: #DC2626; padding: 1px 8px;
  border-radius: 3px; font-size: 10px; font-weight: 600; display: inline-block; }

/* ── Architecture ─────────────────────────────────── */
.arch-pipeline {
  display: flex; flex-direction: column; align-items: center; gap: 0;
  padding: 4px 0;
}
.arch-card {
  border-radius: 8px; border: 1px solid;
  padding: 11px 22px; width: 280px; text-align: center;
}
.arch-card-title { font-size: 13px; font-weight: 600; margin-bottom: 3px; }
.arch-card-sub   { font-size: 11px; }
.arch-arrow-down { font-size: 16px; color: #D1D5DB; line-height: 1.6; text-align: center; }

.arch-data { background: #EFF6FF; border-color: #BFDBFE; }
.arch-data .arch-card-title { color: #1D4ED8; }
.arch-data .arch-card-sub   { color: #3B82F6; }

.arch-proc { background: #F9FAFB; border-color: #E5E7EB; }
.arch-proc .arch-card-title { color: #374151; }
.arch-proc .arch-card-sub   { color: #6B7280; }

.arch-ml { background: #F5F3FF; border-color: #DDD6FE; }
.arch-ml .arch-card-title { color: #5B21B6; }
.arch-ml .arch-card-sub   { color: #7C3AED; }

.arch-out { background: #ECFDF5; border-color: #A7F3D0; }
.arch-out .arch-card-title { color: #065F46; }
.arch-out .arch-card-sub   { color: #059669; }

/* ── Delta cards (scenario) ───────────────────────── */
.delta-card {
  background: #FFFFFF; border: 1px solid #E5E7EB;
  border-radius: 8px; padding: 18px 20px;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.delta-val  { font-size: 26px; font-weight: 700; color: #111827; }
.delta-diff { font-size: 12px; font-weight: 500; margin-top: 3px; }
.delta-pos  { color: #059669; }
.delta-neg  { color: #DC2626; }
.delta-neu  { color: #6B7280; }
.delta-lbl  { font-size: 12px; color: #6B7280; margin-top: 6px; }
.delta-base { font-size: 11px; color: #9CA3AF; margin-top: 2px; }

/* ── Rec panel ─────────────────────────────────────── */
.rec-panel {
  background: #FFFFFF; border: 1px solid #E5E7EB;
  border-radius: 8px; padding: 20px 22px;
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.rec-panel-title {
  font-size: 13px; font-weight: 600; color: #111827; margin-bottom: 14px;
}
.rec-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 9px 0; border-bottom: 1px solid #F3F4F6;
}
.rec-row:last-child { border-bottom: none; }
.rec-ch  { font-size: 13px; font-weight: 500; color: #374151; }
.rec-val { font-size: 13px; font-weight: 600; color: #059669; }

/* ── Divider ───────────────────────────────────────── */
.divider { border-top: 1px solid #E5E7EB; margin: 20px 0; }

/* ── Upload zones ─────────────────────────────────── */
.upload-zone {
  border-radius: 8px; padding: 14px 16px; margin-bottom: 4px;
  border: 1.5px dashed; background: #FAFAFA;
}
.uz-google { border-color: #4285F4; }
.uz-meta   { border-color: #1877F2; }
.uz-bing   { border-color: #008373; }
.uz-title  { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
.uz-google .uz-title { color: #4285F4; }
.uz-meta   .uz-title { color: #1877F2; }
.uz-bing   .uz-title { color: #008373; }

/* ── Table base ───────────────────────────────────── */
.pmie-table {
  width: 100%; border-collapse: collapse;
  font-family: 'Inter', sans-serif; font-size: 12px; color: #374151;
}
.pmie-table th {
  padding: 8px 12px; text-align: left;
  font-size: 10px; font-weight: 600; color: #6B7280;
  text-transform: uppercase; letter-spacing: 0.6px;
  border-bottom: 2px solid #E5E7EB; background: #F9FAFB;
}
.pmie-table td {
  padding: 9px 12px; border-bottom: 1px solid #F3F4F6;
}
.pmie-table tr:last-child td { border-bottom: none; }
.pmie-table tr:hover td { background: #F9FAFB; }

/* ── Tabs (keep Streamlit tabs styled cleanly) ──────── */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #E5E7EB; }
.stTabs [data-baseweb="tab"] {
  font-size: 13px; font-weight: 500; color: #6B7280;
  padding: 10px 16px; border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] { color: #4F46E5 !important; border-bottom-color: #4F46E5 !important; }

/* ── Spinner ──────────────────────────────────────── */
.stSpinner { color: #4F46E5 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
DATA_DIR   = "../data"
MODEL_PATH = "../pickle/model.pkl"
if not os.path.exists(DATA_DIR):   DATA_DIR   = "./data"
if not os.path.exists(MODEL_PATH): MODEL_PATH = "./pickle/model.pkl"

# ─────────────────────────────────────────────────────────────
# CHANNEL COLORS  (official brand)
# ─────────────────────────────────────────────────────────────
CHANNEL_COLORS = {
    'google': '#4285F4',
    'meta':   '#1877F2',
    'bing':   '#008373',
}

# ─────────────────────────────────────────────────────────────
# HELPERS  ——  ALL UNCHANGED
# ─────────────────────────────────────────────────────────────
def fmt_dollar(v):
    if v >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if v >= 1_000:     return f"${v/1_000:.1f}K"
    return f"${v:,.0f}"

def fmt_roas(v): return f"{v:.2f}x"

def confidence_label(pct_float):
    pct = pct_float * 100 if pct_float <= 1.0 else pct_float
    if pct >= 90: return "Very High", "success"
    if pct >= 75: return "High",      "success"
    if pct >= 60: return "Moderate",  "warning"
    return               "Low",       "danger"

def shap_cat_class(cat: str) -> str:
    if "Spend"      in cat: return "cat-spend"
    if "Efficiency" in cat: return "cat-eff"
    return "cat-seas"

def best_channel(df_chan):
    if df_chan.empty: return "N/A"
    return df_chan.sort_values("revenue", ascending=False).iloc[0]["channel"].capitalize()

def top_risk(report):
    risks = [r for r in report.get("risk_radar", []) if r.get("severity") in ("High", "Medium")]
    return risks[0]["risk"] if risks else "None detected"

def top_opp(report):
    opps = report.get("opportunities", [])
    return opps[0][:80] + "…" if opps else "No opportunities identified"

def trend_label(roas):
    if roas >= 3.5: return "Strong", "kpi-delta-up",   "↑"
    if roas >= 2.5: return "Stable", "kpi-delta-flat", "→"
    return                 "Weak",   "kpi-delta-down",  "↓"

def channel_colors_for(df_chan):
    return [CHANNEL_COLORS.get(ch.lower(), '#9CA3AF') for ch in df_chan['channel']]

# ─────────────────────────────────────────────────────────────
# CACHING  ——  UNCHANGED
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_cached_forecast(g_change, m_change, b_change):
    sim_res = run_budget_simulation(DATA_DIR, MODEL_PATH,
                                    google_change=g_change,
                                    meta_change=m_change,
                                    bing_change=b_change)
    daily  = preprocess_all(DATA_DIR)
    weekly = aggregate_to_weekly(daily)

    det_preds     = sim_res['predictions']
    det_total_row = det_preds[(det_preds['horizon'] == '30d') & (det_preds['level'] == 'total')]
    det_30d_spend = float(det_total_row['spend'].values[0]) if len(det_total_row) > 0 else None

    mc_kwargs = {'horizon_weeks': 4}
    if 'deterministic_total_spend' in inspect.signature(run_portfolio_monte_carlo).parameters:
        mc_kwargs['deterministic_total_spend'] = det_30d_spend

    mc_res = run_portfolio_monte_carlo(
        weekly,
        {'google': g_change, 'meta': m_change, 'bing': b_change},
        **mc_kwargs
    )
    shap_str  = compute_shap_explanations(DATA_DIR, MODEL_PATH)
    shap_dict = json.loads(shap_str)
    report    = generate_analyst_report(DATA_DIR, MODEL_PATH, sim_res, shap_dict, mc_res)

    return sim_res, mc_res, shap_dict, report, weekly

@st.cache_data(show_spinner=False)
def get_response_curves():
    daily  = preprocess_all(DATA_DIR)
    weekly = aggregate_to_weekly(daily)
    return fit_response_curves(weekly), weekly

# ─────────────────────────────────────────────────────────────
# CHART DEFAULTS  —  light theme
# ─────────────────────────────────────────────────────────────
CHART_THEME = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_family="Inter, Segoe UI, sans-serif",
    font_color="#374151",
    margin=dict(l=12, r=12, t=40, b=12),
)

def _apply_light_axes(fig):
    fig.update_xaxes(
        gridcolor="#F3F4F6", linecolor="#E5E7EB",
        tickfont=dict(size=11, color="#6B7280"),
        title_font=dict(size=12, color="#374151"),
    )
    fig.update_yaxes(
        gridcolor="#F3F4F6", linecolor="#E5E7EB",
        tickfont=dict(size=11, color="#6B7280"),
        title_font=dict(size=12, color="#374151"),
    )
    return fig

# ─────────────────────────────────────────────────────────────
# AI MEMO RENDERER  —  clean SaaS sections
# ─────────────────────────────────────────────────────────────
def render_ai_memo(report):
    def esc(t):
        return str(t).replace('$', r'\$') if isinstance(t, str) else str(t)

    status = report.get('llm_status', 'unknown')
    if   status == 'generated':     st.caption("AI memo · Gemini 2.5 Flash")
    elif status == 'fallback':
        st.warning("Gemini unavailable — data-derived memo displayed.")
        with st.expander("Error details"):
            st.code(report.get('llm_error') or "No details.")
    elif status == 'not_configured': st.info("GEMINI_API_KEY not set — data-derived memo displayed.")

    # Executive Summary
    exec_sum = esc(report.get('executive_summary', ''))
    st.markdown(f"""
    <div class="memo-section">
      <div class="memo-title">
        <span class="memo-icon mi-indigo">📋</span>Executive Summary
      </div>
      <div class="memo-body">{exec_sum}</div>
    </div>""", unsafe_allow_html=True)

    # Opportunities
    opps = report.get('opportunities', [])
    if opps:
        items = "".join(
            f'<div class="memo-item"><span style="color:#059669;margin-top:2px;">›</span>'
            f'<span>{esc(o)}</span></div>' for o in opps
        )
        st.markdown(f"""
        <div class="memo-section">
          <div class="memo-title"><span class="memo-icon mi-green">💡</span>Opportunities</div>
          {items}
        </div>""", unsafe_allow_html=True)

    # Channel Insights
    chan_ins = report.get('channel_insights', [])
    if chan_ins:
        items = "".join(
            f'<div class="memo-item"><span style="color:#2563EB;margin-top:2px;">›</span>'
            f'<span>{esc(c)}</span></div>' for c in chan_ins
        )
        st.markdown(f"""
        <div class="memo-section">
          <div class="memo-title"><span class="memo-icon mi-blue">📊</span>Channel Insights</div>
          {items}
        </div>""", unsafe_allow_html=True)

    # Risk Radar
    risks = report.get('risk_radar', [])
    if risks:
        items = ""
        for r in risks:
            sev = r.get('severity', 'Low')
            bcls = "mbr-high" if sev == "High" else ("mbr-medium" if sev == "Medium" else "mbr-low")
            items += (
                f'<div class="memo-item">'
                f'<span class="memo-risk-badge {bcls}">{sev}</span>'
                f'<span><b>{esc(r.get("risk",""))}</b> — {esc(r.get("mitigation",""))}</span>'
                f'</div>'
            )
        st.markdown(f"""
        <div class="memo-section">
          <div class="memo-title"><span class="memo-icon mi-red">⚠</span>Risk Radar</div>
          {items}
        </div>""", unsafe_allow_html=True)

    # Budget Recommendations
    recs = report.get('budget_recommendations', [])
    if recs:
        items = "".join(
            f'<div class="memo-item"><span style="color:#D97706;margin-top:2px;">›</span>'
            f'<span>{esc(rec)}</span></div>' for rec in recs
        )
        st.markdown(f"""
        <div class="memo-section">
          <div class="memo-title"><span class="memo-icon mi-amber">💰</span>Budget Recommendations</div>
          {items}
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR  —  navigation
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 16px 12px 16px; border-bottom:1px solid #1E293B; margin-bottom:8px;">
      <div style="font-size:16px;font-weight:700;color:#F1F5F9;letter-spacing:-0.3px;">⚡ PMIE</div>
      <div style="font-size:11px;color:#64748B;margin-top:3px;">Probabilistic Marketing Intelligence Engine</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        options=[
            "Forecast Dashboard",
            "Scenario Builder",
            "Response Curves",
            "Data Ingestion",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<hr>', unsafe_allow_html=True)

    model_exists = os.path.exists(MODEL_PATH)
    model_dot    = "dot-green" if model_exists else "dot-red"
    model_text   = "Ready" if model_exists else "Not found"

    st.markdown(f"""
    <div style="padding:0 12px;">
      <div style="font-size:10px;font-weight:600;color:#475569;text-transform:uppercase;
                  letter-spacing:0.8px;margin-bottom:10px;">System Status</div>
      <div style="font-size:12px;color:#94A3B8;margin-bottom:6px;">
        <span class="status-dot {model_dot}" style="display:inline-block;width:7px;height:7px;
              border-radius:50%;margin-right:5px;background:{'#10B981' if model_exists else '#EF4444'};"></span>
        Model {model_text}
      </div>
      <div style="font-size:11px;color:#475569;">Data: {DATA_DIR}</div>
      <div style="font-size:11px;color:#475569;margin-top:12px;">v2.0  ·  LightGBM + Conformal</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOP BAR  (shown on all pages)
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <div class="top-bar-left">
    <span class="top-bar-logo">PMIE</span>
    <span class="top-bar-sub">Revenue Forecasting &nbsp;·&nbsp; Scenario Planning &nbsp;·&nbsp; AI Insights</span>
  </div>
  <div class="top-bar-pills">
    <span class="pill">LightGBM</span>
    <span class="pill">Conformal</span>
    <span class="pill">Monte Carlo</span>
    <span class="pill">SHAP</span>
    <span class="pill pill-green">Gemini</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE: FORECAST DASHBOARD
# ═════════════════════════════════════════════════════════════
if page == "Forecast Dashboard":
    df_pred_global = pd.DataFrame()   # safe default if model absent

    if not os.path.exists(MODEL_PATH):
        st.error(f"Trained model not found at `{MODEL_PATH}`. Run training scripts first.")
    else:
        with st.spinner("Loading forecasts…"):
            sim_res, mc_res, shap_dict, report, weekly_df = get_cached_forecast(0.0, 0.0, 0.0)

        df_pred        = sim_res['predictions']
        df_pred_global = df_pred.copy()

        # Horizon selector — compact inline
        hz_col, _ = st.columns([3, 7])
        with hz_col:
            horizon = st.radio(
                "Forecast Horizon",
                ["30d", "60d", "90d"],
                horizontal=True,
                label_visibility="collapsed",
            )

        df_total = df_pred[(df_pred['horizon'] == horizon) & (df_pred['level'] == 'total')]
        df_chan  = df_pred[(df_pred['horizon'] == horizon) & (df_pred['level'] == 'channel')]

        exp_rev   = float(df_total['revenue'].values[0])
        exp_roas  = float(df_total['roas'].values[0])
        exp_spend = float(df_total['spend'].values[0])
        exp_p10   = float(df_total['revenue_p10'].values[0])
        exp_p90   = float(df_total['revenue_p90'].values[0])
        conf_raw  = float(df_pred['confidence_score'].mean())
        conf_pct  = conf_raw * 100
        conf_lbl, conf_css = confidence_label(conf_raw)

        mc_total  = mc_res.get('total', {})
        mc_p10    = mc_total.get('revenue_p10', exp_p10)
        mc_p50    = mc_total.get('revenue_p50', exp_rev)
        mc_p90    = mc_total.get('revenue_p90', exp_p90)
        mc_roas   = mc_total.get('expected_roas', exp_roas)
        mc_exp_rv = mc_total.get('expected_revenue', mc_p50)

        best_ch = best_channel(df_chan)
        trend_txt, trend_cls, trend_arrow = trend_label(exp_roas)
        conf_css_val = "kpi-value-success" if conf_pct >= 75 else ("kpi-value-warning" if conf_pct >= 60 else "kpi-value-danger")

        # ── EXECUTIVE BRIEF (hero) ──────────────────────────────────────────
        exec_sum     = report.get('executive_summary', '')
        exec_esc     = exec_sum.replace('$', r'\$')
        first_sent   = exec_esc.split('.')[0].strip()
        if len(first_sent) > 110: first_sent = first_sent[:107] + "…"

        risk_level   = "High" if any(r.get('severity') == 'High' for r in report.get('risk_radar', [])) else \
                       ("Medium" if any(r.get('severity') == 'Medium' for r in report.get('risk_radar', [])) else "Low")
        risk_chip_cls = "chip-red" if risk_level == "High" else ("chip-amber" if risk_level == "Medium" else "chip-green")

        risks_items = "".join(
            f'<div class="exec-item">› {r.get("risk","")[:70]}</div>'
            for r in report.get('risk_radar', [])[:3]
        ) or '<div class="exec-item" style="color:#9CA3AF;">No significant risks</div>'

        opps_items = "".join(
            f'<div class="exec-item">› {o[:70]}</div>'
            for o in report.get('opportunities', [])[:3]
        ) or '<div class="exec-item" style="color:#9CA3AF;">No opportunities identified</div>'

        body_text = exec_esc[:320] + ("…" if len(exec_esc) > 320 else "")

        st.markdown(f"""
        <div class="exec-brief">
          <div class="exec-headline">{first_sent}</div>
          <div class="exec-body">{body_text}</div>
          <div class="exec-chips">
            <span class="exec-chip chip-indigo">Revenue: {fmt_dollar(exp_rev)}</span>
            <span class="exec-chip chip-green">Confidence: {conf_pct:.0f}%</span>
            <span class="exec-chip {risk_chip_cls}">Risk: {risk_level}</span>
            <span class="exec-chip chip-blue">Best Channel: {best_ch}</span>
          </div>
          <div style="display:flex;gap:24px;">
            <div style="flex:1;">
              <div class="exec-col-title">Key Risks</div>
              {risks_items}
            </div>
            <div style="flex:1;">
              <div class="exec-col-title">Opportunities</div>
              {opps_items}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI GRID — Row 1 ────────────────────────────────────────────────
        st.markdown('<div class="section-title">Performance Forecast · ' + horizon + '</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)

        def kpi_card(col, label, value, css_cls="", delta_text="", delta_cls=""):
            delta_html = f'<div class="kpi-delta {delta_cls}">{delta_text}</div>' if delta_text else ""
            with col:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-label">{label}</div>
                  <div class="kpi-value {css_cls}">{value}</div>
                  {delta_html}
                </div>""", unsafe_allow_html=True)

        kpi_card(k1, "Expected Revenue",  fmt_dollar(exp_rev),   "",                    f"{trend_arrow} {trend_txt}", trend_cls)
        kpi_card(k2, "Forecast ROAS",     fmt_roas(exp_roas),    "kpi-value-primary",   f"{trend_arrow} {trend_txt}", trend_cls)
        kpi_card(k3, "Confidence Score",  f"{conf_pct:.1f}%",    conf_css_val,          conf_lbl)
        kpi_card(k4, "Best Channel",      best_ch,               "kpi-value-primary")

        st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

        # Row 2
        k5, k6, k7, k8 = st.columns(4)
        kpi_card(k5, "P10 — Downside",   fmt_dollar(mc_p10),   "kpi-value-danger")
        kpi_card(k6, "P50 — Median",     fmt_dollar(mc_p50),   "kpi-value-warning")
        kpi_card(k7, "P90 — Upside",     fmt_dollar(mc_p90),   "kpi-value-success")
        kpi_card(k8, "Planned Spend",    fmt_dollar(exp_spend), "")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── FORECAST RELIABILITY ────────────────────────────────────────────
        anoms      = report.get('anomaly_detection', {}).get('recent_anomalies', [])
        drift_data = report.get('data_drift', {})
        high_drifts = [k for k, v in drift_data.items() if v.get('status') == 'High Drift']
        mod_drifts  = [k for k, v in drift_data.items() if v.get('status') == 'Moderate Drift']

        anom_val  = f"{len(anoms)} flagged" if anoms else "Clean"
        drift_val = "High" if high_drifts else ("Moderate" if mod_drifts else "Stable")
        anom_dot  = "dot-amber" if anoms else "dot-green"
        drift_dot = "dot-red" if high_drifts else ("dot-amber" if mod_drifts else "dot-green")
        anom_bg   = "#F59E0B" if anoms else "#10B981"
        drift_bg  = "#EF4444" if high_drifts else ("#F59E0B" if mod_drifts else "#10B981")

        st.markdown('<div class="section-title">Forecast Reliability</div>', unsafe_allow_html=True)

        rel_left, rel_right = st.columns([2, 1])

        tiles = [
            ("MAPE",          "2.95%",    "Portfolio accuracy",           "#10B981"),
            ("Coverage",      "85–90%",   "Conformal intervals",          "#10B981"),
            ("MC Simulations","10,000",   "Gaussian Copula",              "#10B981"),
            ("Anomalies",     anom_val,   "Last 90 days",                 anom_bg),
            ("Drift",         drift_val,  "PSI monitoring",               drift_bg),
            ("Backtest",      "R²=0.98",  "Cross-validated",              "#10B981"),
        ]

        with rel_left:
            for i in range(0, 6, 2):
                col_a, col_b = st.columns(2)
                for col, (title, value, sub, dot_color) in zip([col_a, col_b], tiles[i:i+2]):
                    with col:
                        st.markdown(f"""
                        <div class="rel-tile">
                          <div class="rel-tile-title">
                            <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                                         background:{dot_color};margin-right:5px;"></span>{title}
                          </div>
                          <div class="rel-tile-value">{value}</div>
                          <div class="rel-tile-sub">{sub}</div>
                        </div>""", unsafe_allow_html=True)

        with rel_right:
            gauge_color = "#059669" if conf_pct >= 75 else ("#D97706" if conf_pct >= 60 else "#DC2626")
            status_text = "High" if conf_pct >= 75 else ("Moderate" if conf_pct >= 60 else "Low")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=conf_pct,
                number={"suffix": "%", "font": {"size": 36, "family": "Inter", "color": gauge_color}},
                gauge={
                    "axis": {"range": [0, 100], "tickfont": {"size": 10, "color": "#9CA3AF"}},
                    "bar": {"color": gauge_color, "thickness": 0.22},
                    "bgcolor": "#F9FAFB",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 60],   "color": "#FEE2E2"},
                        {"range": [60, 75],  "color": "#FEF3C7"},
                        {"range": [75, 90],  "color": "#D1FAE5"},
                        {"range": [90, 100], "color": "#A7F3D0"},
                    ],
                    "threshold": {"line": {"color": gauge_color, "width": 2}, "thickness": 0.8, "value": conf_pct},
                },
            ))
            fig_gauge.update_layout(
                height=260, margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)", font_family="Inter",
            )
            gauge_html = pio.to_html(fig_gauge, full_html=False, include_plotlyjs='cdn', config={'displayModeBar': False})
            
            full_html = f"""
            <html>
            <head>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
            <style>
            body {{ margin: 0; font-family: 'Inter', sans-serif; }}
            .reliability-gauge-card {{
                background: white;
                border: 1px solid #E5E7EB;
                border-radius: 16px;
                padding: 24px;
                height: 452px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                text-align: center;
                box-sizing: border-box;
                box-shadow: 0 1px 3px rgba(0,0,0,.05);
            }}
            .rg-title {{ font-size: 15px; font-weight: 600; color: #111827; margin-bottom: 6px; }}
            .rg-sub {{ font-size: 13px; color: #6B7280; margin-bottom: 0px; }}
            .rg-badge {{ margin-top: -5px; font-size: 13px; font-weight: 600; color: {gauge_color}; }}
            </style>
            </head>
            <body>
            <div class="reliability-gauge-card">
                <div class="rg-title">Forecast Confidence</div>
                <div class="rg-sub">Expected Revenue Range<br><b style="color:#374151;font-size:14px;">{fmt_dollar(exp_p10)} – {fmt_dollar(exp_p90)}</b></div>
                <div style="width: 100%; flex: 1; display:flex; align-items:center; justify-content:center;">
                    {gauge_html}
                </div>
                <div class="rg-badge">
                    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{gauge_color};margin-right:6px;"></span>
                    {status_text} Confidence
                </div>
            </div>
            </body>
            </html>
            """
            components.html(full_html, height=456)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── CHANNEL FORECAST CHARTS ─────────────────────────────────────────
        st.markdown('<div class="section-title">Channel Forecast Breakdown</div>', unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)

        with cc1:
            fig_rev = go.Figure()
            for _, row in df_chan.iterrows():
                color = CHANNEL_COLORS.get(row['channel'].lower(), '#9CA3AF')
                fig_rev.add_trace(go.Bar(
                    name=row['channel'].capitalize(),
                    x=[row['channel'].capitalize()],
                    y=[row['revenue']],
                    marker_color=color,
                    error_y=dict(type='data', symmetric=False,
                                 array=[row['revenue_p90'] - row['revenue']],
                                 arrayminus=[row['revenue'] - row['revenue_p10']],
                                 color='#D1D5DB'),
                    showlegend=True,
                ))
            fig_rev.update_layout(title=f"Revenue by Channel ({horizon})",
                                  yaxis_title="Revenue ($)", barmode='group', **CHART_THEME)
            _apply_light_axes(fig_rev)
            st.plotly_chart(fig_rev, use_container_width=True)

        with cc2:
            fig_roas_ch = go.Figure()
            for _, row in df_chan.iterrows():
                color = CHANNEL_COLORS.get(row['channel'].lower(), '#9CA3AF')
                fig_roas_ch.add_trace(go.Bar(
                    name=row['channel'].capitalize(),
                    x=[row['channel'].capitalize()],
                    y=[row['roas']],
                    marker_color=color,
                    error_y=dict(type='data', symmetric=False,
                                 array=[row['roas_p90'] - row['roas']],
                                 arrayminus=[row['roas'] - row['roas_p10']],
                                 color='#D1D5DB'),
                    showlegend=True,
                ))
            fig_roas_ch.update_layout(title=f"ROAS by Channel ({horizon})",
                                      yaxis_title="ROAS (×)", barmode='group', **CHART_THEME)
            _apply_light_axes(fig_roas_ch)
            st.plotly_chart(fig_roas_ch, use_container_width=True)

        cc3, cc4 = st.columns(2)
        with cc3:
            channel_names = df_chan['channel'].str.capitalize().tolist()
            donut_colors  = [CHANNEL_COLORS.get(ch.lower(), '#9CA3AF') for ch in df_chan['channel']]
            fig_donut = px.pie(df_chan, values='revenue', names=channel_names,
                               hole=0.45, color_discrete_sequence=donut_colors)
            fig_donut.update_layout(title="Revenue Mix", **CHART_THEME)
            fig_donut.update_traces(textfont_size=12, textfont_color="#374151")
            st.plotly_chart(fig_donut, use_container_width=True)

        with cc4:
            df_type = df_pred[(df_pred['horizon'] == horizon) & (df_pred['level'] == 'campaign_type')]
            fig_type = px.bar(
                df_type, x='campaign_type', y='revenue', color='channel',
                labels={'revenue': 'Revenue ($)', 'campaign_type': 'Campaign Type'},
                title=f"Revenue by Campaign Type ({horizon})",
                color_discrete_map={ch.capitalize(): col for ch, col in CHANNEL_COLORS.items()},
            )
            fig_type.update_layout(**CHART_THEME)
            _apply_light_axes(fig_type)
            st.plotly_chart(fig_type, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── MONTE CARLO ─────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Monte Carlo Portfolio Simulation</div>', unsafe_allow_html=True)

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc_cards = [
            (mc1, "Worst Case (P10)",    fmt_dollar(mc_p10),    "mc-worst",  "10th percentile · Conservative floor"),
            (mc2, "Most Likely (P50)",   fmt_dollar(mc_p50),    "mc-likely", "Median · Most probable"),
            (mc3, "Best Case (P90)",     fmt_dollar(mc_p90),    "mc-best",   "90th percentile · Optimistic ceiling"),
            (mc4, "Expected Revenue",    fmt_dollar(mc_exp_rv), "mc-exp",    "Probability-weighted mean"),
            (mc5, "MC ROAS",             fmt_roas(mc_roas),     "mc-roas",   "Portfolio return on ad spend"),
        ]
        for col, lbl, val, vcls, sub in mc_cards:
            with col:
                st.markdown(f"""
                <div class="mc-card">
                  <div class="mc-label">{lbl}</div>
                  <div class="mc-value {vcls}">{val}</div>
                  <div class="mc-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        hist_bins  = mc_total.get('histogram_bins', [])
        hist_edges = mc_total.get('histogram_edges', [])
        if hist_bins and hist_edges:
            midpoints = [(hist_edges[i] + hist_edges[i+1]) / 2 for i in range(len(hist_bins))]
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Bar(
                x=midpoints, y=hist_bins,
                marker_color='#4F46E5', marker_opacity=0.7, name="Simulations",
            ))
            fig_hist.add_vline(x=mc_total.get('expected_revenue', mc_p50),
                line_width=2, line_dash="dash", line_color="#374151",
                annotation_text="Expected", annotation_font_size=10, annotation_font_color="#374151")
            fig_hist.add_vline(x=mc_p10, line_width=1, line_dash="dot",
                line_color="#DC2626", annotation_text="P10", annotation_font_size=10)
            fig_hist.add_vline(x=mc_p90, line_width=1, line_dash="dot",
                line_color="#059669", annotation_text="P90", annotation_font_size=10)
            fig_hist.update_layout(
                title="10,000 Revenue Simulations — Empirical Copula",
                xaxis_title="Revenue ($)", yaxis_title="Frequency",
                **CHART_THEME,
            )
            _apply_light_axes(fig_hist)
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── SHAP ────────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Why the Model Predicts This — SHAP Drivers</div>', unsafe_allow_html=True)
        shap_c1, shap_c2 = st.columns([2, 3])

        with shap_c1:
            rev_drivers = shap_dict.get('revenue', {}).get('drivers', [])
            if rev_drivers:
                cat_config = {
                    "Spend Drivers": {
                        "card_cls": "driver-card-spend", "badge_cls": "badge-spend",
                        "title": "Spend",
                        "desc": "Media investment levels directly drive the predicted revenue trajectory.",
                    },
                    "Efficiency Drivers": {
                        "card_cls": "driver-card-eff", "badge_cls": "badge-eff",
                        "title": "Efficiency",
                        "desc": "Historical ROAS trends shape expected return per dollar of spend.",
                    },
                    "Seasonality Drivers": {
                        "card_cls": "driver-card-seas", "badge_cls": "badge-seas",
                        "title": "Seasonality",
                        "desc": "Time-based patterns captured via lag and rolling window features.",
                    },
                }
                cat_top = {}
                for d in rev_drivers[:5]:
                    cat = d.get('category', '')
                    if cat not in cat_top:
                        cat_top[cat] = d.get('feature', '').replace('_', ' ').title()

                for cat_key, cfg in cat_config.items():
                    top_feat = cat_top.get(cat_key, "—")
                    st.markdown(f"""
                    <div class="driver-card {cfg['card_cls']}">
                      <div class="driver-title">
                        {cfg['title']}
                        <span class="driver-badge {cfg['badge_cls']}">{cfg['title']}</span>
                      </div>
                      <div class="driver-feat">Top feature: {top_feat}</div>
                      <div class="driver-desc">{cfg['desc']}</div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("SHAP drivers not available.")

        with shap_c2:
            if rev_drivers:
                df_drivers = pd.DataFrame(rev_drivers)
                fig_shap = px.bar(
                    df_drivers, x='importance', y='feature', color='category',
                    orientation='h',
                    labels={'importance': 'Mean |SHAP|', 'feature': ''},
                    color_discrete_map={
                        'Spend Drivers':       '#4F46E5',
                        'Efficiency Drivers':  '#7C3AED',
                        'Seasonality Drivers': '#D97706',
                    },
                )
                fig_shap.update_layout(
                    title="Top Revenue Forecast Drivers",
                    yaxis={'categoryorder': 'total ascending'},
                    legend_title="Category",
                    **CHART_THEME,
                )
                _apply_light_axes(fig_shap)
                st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── AI MEMO ─────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">AI Marketing Analyst</div>', unsafe_allow_html=True)
        render_ai_memo(report)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # ── DIAGNOSTICS ──────────────────────────────────────────────────────
        with st.expander("Diagnostics — Anomaly Log & Drift Monitoring"):
            dcol1, dcol2 = st.columns(2)

            with dcol1:
                st.markdown("**Anomaly Log (Last 90 Days)**")
                anoms_diag = report.get('anomaly_detection', {}).get('recent_anomalies', [])
                if anoms_diag:
                    rows = ""
                    for a in anoms_diag:
                        dev = abs(float(a.get('deviation_pct', 0)))
                        if dev > 30:   sev, scls = "Critical", "sev-critical"
                        elif dev > 15: sev, scls = "High",     "sev-high"
                        elif dev > 5:  sev, scls = "Medium",   "sev-medium"
                        else:          sev, scls = "Low",       "sev-low"
                        dv = a.get('deviation_pct', 0)
                        dv_str = f"{dv:+.1f}%" if isinstance(dv, (int, float)) else str(dv)
                        dv_color = "#DC2626" if isinstance(dv, (int, float)) and dv < 0 else "#059669"
                        rows += f"""<tr>
                          <td>{a.get('date','')}</td>
                          <td>${float(a.get('actual',0)):,.0f}</td>
                          <td>${float(a.get('expected',0)):,.0f}</td>
                          <td style="color:{dv_color};">{dv_str}</td>
                          <td><span class="sev-badge {scls}">{sev}</span></td>
                        </tr>"""
                    st.markdown(f"""
                    <table class="pmie-table">
                      <thead><tr>
                        <th>Date</th><th>Actual</th><th>Expected</th><th>Deviation</th><th>Severity</th>
                      </tr></thead>
                      <tbody>{rows}</tbody>
                    </table>""", unsafe_allow_html=True)
                else:
                    st.success("No anomalies detected in the last 90 days.")

            with dcol2:
                st.markdown("**Drift Monitoring (PSI)**")
                drift_list = [
                    {'Indicator': m.upper(), 'PSI': v['psi'], 'Status': v['status']}
                    for m, v in report.get('data_drift', {}).items()
                ]
                if drift_list:
                    rows = ""
                    for d in drift_list:
                        psi = float(d['PSI'])
                        status = d['Status']
                        if status == 'High Drift' or psi > 0.25:
                            pcls = "psi-high-drift"
                        elif status == 'Moderate Drift' or psi > 0.1:
                            pcls = "psi-moderate"
                        else:
                            pcls = "psi-stable"
                        rows += f"""<tr>
                          <td><b>{d['Indicator']}</b></td>
                          <td>{psi:.3f}</td>
                          <td><span class="{pcls}">{status}</span></td>
                        </tr>"""
                    st.markdown(f"""
                    <table class="pmie-table">
                      <thead><tr>
                        <th>Indicator</th><th>PSI Value</th><th>Status</th>
                      </tr></thead>
                      <tbody>{rows}</tbody>
                    </table>""", unsafe_allow_html=True)
                else:
                    st.info("No drift data available.")

        # ── ARCHITECTURE ────────────────────────────────────────────────────
        with st.expander("How PMIE Works — Architecture"):
            arch_col, _ = st.columns([1, 1])
            with arch_col:
                steps = [
                    ("Data Sources",         "Google Ads · Meta Ads · Bing Ads",            "arch-data"),
                    ("Preprocessing",        "Date parsing · Budget normalization",          "arch-proc"),
                    ("Weekly Aggregation",   "Campaign-week summaries · Channel rollups",   "arch-proc"),
                    ("Feature Engineering",  "Lags · Rolling averages · Horizon encoding",  "arch-proc"),
                    ("LightGBM Forecast",    "3 models: 30-day · 60-day · 90-day revenue", "arch-ml"),
                    ("Conformal Prediction", "Split-conformal P10/P90 intervals",           "arch-ml"),
                    ("Monte Carlo Sim",      "10,000 Gaussian Copula simulations",          "arch-ml"),
                    ("SHAP Explainability",  "Top-5 revenue drivers by importance",         "arch-ml"),
                    ("Gemini Analyst",       "LLM-generated executive memo",                "arch-ml"),
                    ("Dashboard",            "Probabilistic KPIs · Scenarios · Alerts",     "arch-out"),
                ]

                arch_html = '<div class="arch-pipeline">'
                for i, (label, detail, cls) in enumerate(steps):
                    is_last = i == len(steps) - 1
                    arch_html += f"""
                    <div class="arch-card {cls}">
                      <div class="arch-card-title">{label}</div>
                      <div class="arch-card-sub">{detail}</div>
                    </div>"""
                    if not is_last:
                        arch_html += '<div class="arch-arrow-down">↓</div>'
                arch_html += '</div>'
                st.markdown(arch_html, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# PAGE: SCENARIO BUILDER
# ═════════════════════════════════════════════════════════════
elif page == "Scenario Builder":
    st.markdown('<div class="section-title">Budget Scenario Planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Adjust spend by channel to simulate impact on revenue and ROAS.</div>', unsafe_allow_html=True)

    col_adj1, col_adj2, col_adj3 = st.columns(3)
    with col_adj1: g_adj = st.slider("Google Ads Spend Shift (%)", -50, 100, 0, 5) / 100.0
    with col_adj2: m_adj = st.slider("Meta Ads Spend Shift (%)",   -50, 100, 0, 5) / 100.0
    with col_adj3: b_adj = st.slider("Bing Ads Spend Shift (%)",   -50, 100, 0, 5) / 100.0

    with st.spinner("Running scenario…"):
        sim_res_s, mc_res_s, shap_dict_s, report_s, _ = get_cached_forecast(g_adj, m_adj, b_adj)
        # Baseline for comparison
        sim_res_b, _, _, _, _ = get_cached_forecast(0.0, 0.0, 0.0)

    df_sim = sim_res_s['predictions']
    df_base_global = sim_res_b['predictions']

    sim_horizon = st.radio("Simulation Horizon:", ["30d", "60d", "90d"], key="sim_h", horizontal=True)

    df_base_total = df_base_global[(df_base_global['horizon'] == sim_horizon) & (df_base_global['level'] == 'total')]
    df_sim_total  = df_sim[(df_sim['horizon'] == sim_horizon) & (df_sim['level'] == 'total')]

    if df_base_total.empty or df_sim_total.empty:
        st.warning("Baseline or simulation predictions unavailable. Load model first.")
    else:
        rev_base  = float(df_base_total['revenue'].values[0])
        rev_sim   = float(df_sim_total['revenue'].values[0])
        roas_base = float(df_base_total['roas'].values[0])
        roas_sim  = float(df_sim_total['roas'].values[0])
        sp_base   = float(df_base_total['spend'].values[0])
        sp_sim    = float(df_sim_total['spend'].values[0])

        rev_delta  = (rev_sim - rev_base) / rev_base * 100 if rev_base > 0 else 0.0
        roas_delta = roas_sim - roas_base
        sp_delta   = (sp_sim - sp_base) / sp_base * 100 if sp_base > 0 else 0.0

        def ddcls(v): return "delta-pos" if v > 0 else ("delta-neg" if v < 0 else "delta-neu")
        def darr(v):  return "↑" if v > 0 else ("↓" if v < 0 else "→")

        d1, d2, d3 = st.columns(3)
        with d1:
            st.markdown(f"""
            <div class="delta-card">
              <div class="delta-lbl">Expected Revenue</div>
              <div class="delta-val">{fmt_dollar(rev_sim)}</div>
              <div class="delta-diff {ddcls(rev_delta)}">{darr(rev_delta)} {rev_delta:+.1f}% vs baseline</div>
              <div class="delta-base">Baseline: {fmt_dollar(rev_base)}</div>
            </div>""", unsafe_allow_html=True)
        with d2:
            st.markdown(f"""
            <div class="delta-card">
              <div class="delta-lbl">Expected ROAS</div>
              <div class="delta-val">{fmt_roas(roas_sim)}</div>
              <div class="delta-diff {ddcls(roas_delta)}">{darr(roas_delta)} {roas_delta:+.2f}× vs baseline</div>
              <div class="delta-base">Baseline: {fmt_roas(roas_base)}</div>
            </div>""", unsafe_allow_html=True)
        with d3:
            st.markdown(f"""
            <div class="delta-card">
              <div class="delta-lbl">Planned Spend</div>
              <div class="delta-val">{fmt_dollar(sp_sim)}</div>
              <div class="delta-diff {ddcls(sp_delta)}">{darr(sp_delta)} {sp_delta:+.1f}% vs baseline</div>
              <div class="delta-base">Baseline: {fmt_dollar(sp_base)}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        sc1, sc2 = st.columns([3, 2])
        with sc1:
            df_base_ch = df_base_global[(df_base_global['horizon'] == sim_horizon) & (df_base_global['level'] == 'channel')].copy()
            df_sim_ch  = df_sim[(df_sim['horizon'] == sim_horizon) & (df_sim['level'] == 'channel')].copy()
            df_base_ch['Scenario'] = 'Baseline'
            df_sim_ch['Scenario']  = 'Simulated'
            df_comp = pd.concat([df_base_ch, df_sim_ch], ignore_index=True)

            fig_comp = go.Figure()
            for ch_name in df_comp['channel'].unique():
                base_color = CHANNEL_COLORS.get(ch_name.lower(), '#9CA3AF')
                for scenario, opacity in [('Baseline', 0.4), ('Simulated', 1.0)]:
                    sub = df_comp[(df_comp['channel'] == ch_name) & (df_comp['Scenario'] == scenario)]
                    if sub.empty: continue
                    fig_comp.add_trace(go.Bar(
                        name=f"{ch_name.capitalize()} ({scenario})",
                        x=[ch_name.capitalize()],
                        y=sub['revenue'].values,
                        marker_color=base_color,
                        opacity=opacity,
                    ))
            fig_comp.update_layout(title=f"Revenue Impact ({sim_horizon})", barmode='group', **CHART_THEME)
            _apply_light_axes(fig_comp)
            st.plotly_chart(fig_comp, use_container_width=True)

        with sc2:
            if not df_base_ch.empty:
                best_rev_ch     = df_base_ch.sort_values('revenue', ascending=False).iloc[0]
                weakest_roas_ch = df_base_ch.sort_values('roas', ascending=True).iloc[0]
                rec_google = "+20%" if best_rev_ch.get('channel','').lower() == 'google' else "+10%"
                rec_meta   = "+20%" if best_rev_ch.get('channel','').lower() == 'meta'   else "+5%"
                rec_bing   = "-10%" if weakest_roas_ch.get('channel','').lower() == 'bing' else "0%"

                st.markdown(f"""
                <div class="rec-panel">
                  <div class="rec-panel-title">Recommended Allocation</div>
                  <div class="rec-row">
                    <span class="rec-ch" style="color:#4285F4;">Google Ads</span>
                    <span class="rec-val">{rec_google}</span>
                  </div>
                  <div class="rec-row">
                    <span class="rec-ch" style="color:#1877F2;">Meta Ads</span>
                    <span class="rec-val">{rec_meta}</span>
                  </div>
                  <div class="rec-row">
                    <span class="rec-ch" style="color:#008373;">Bing Ads</span>
                    <span class="rec-val">{rec_bing}</span>
                  </div>
                  <div style="margin-top:12px;font-size:11px;color:#9CA3AF;line-height:1.5;">
                    Highest-revenue: <b style="color:#374151">{best_rev_ch.get('channel','').capitalize()}</b>
                    (ROAS {best_rev_ch.get('roas',0):.2f}×) ·
                    Lowest-ROAS: <b style="color:#374151">{weakest_roas_ch.get('channel','').capitalize()}</b>
                    (ROAS {weakest_roas_ch.get('roas',0):.2f}×)
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Channel data unavailable.")

        with st.expander("AI Analyst Memo - This Scenario"):
            render_ai_memo(report_s)

        with st.expander("Key Assumptions & Limitations"):
            st.markdown("**Core Assumptions:**")
            for a in report_s.get('key_assumptions', []):
                st.markdown(f"- {a}")
            st.markdown("**Forecast Limitations:**")
            for lim in report_s.get('forecast_limitations', []):
                st.markdown(f"- {lim}")
            st.markdown("**Confidence Assessment:**")
            st.markdown(report_s.get('confidence_assessment', 'Not available.'))

# ═════════════════════════════════════════════════════════════
# PAGE: RESPONSE CURVES
# ═════════════════════════════════════════════════════════════
elif page == "Response Curves":
    st.markdown('<div class="section-title">Budget Response Curves</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">How expected revenue responds to changes in spend for each campaign type, fitted via a Hill function (Michaelis-Menten saturation model).</div>', unsafe_allow_html=True)

    with st.spinner("Fitting response curves…"):
        curves, weekly_for_curves = get_response_curves()

    RC_COLORS = {'SEARCH': '#4285F4', 'SHOPPING': '#1877F2', 'PMAX': '#008373'}
    campaign_types = ['SEARCH', 'SHOPPING', 'PMAX']

    rc1, rc2, rc3 = st.columns(3)
    rc_cols = {'SEARCH': rc1, 'SHOPPING': rc2, 'PMAX': rc3}

    for c_type in campaign_types:
        with rc_cols[c_type]:
            if c_type not in curves:
                st.info(f"No data for {c_type}")
                continue

            p        = curves[c_type]
            max_obs  = p['max_observed_spend']
            slope    = p['slope']
            Vmax     = p['Vmax']
            K        = p['K']
            n_hill   = p['n']
            fit_ok   = p['fit_success']

            x_max   = max_obs * 2.5
            x_range = np.linspace(0, x_max, 300)
            y_hill  = np.array([
                slope * max_obs * (hill_function(xi, Vmax, K, n_hill) /
                                   hill_function(max_obs, Vmax, K, n_hill))
                if xi > max_obs and hill_function(max_obs, Vmax, K, n_hill) > 0
                else slope * xi
                for xi in x_range
            ])
            y_merged = np.where(x_range <= max_obs, slope * x_range, y_hill)
            y_merged = np.clip(y_merged, 0, None)

            df_wkly = weekly_for_curves.groupby(['campaign_type', 'date'], observed=False).agg(
                spend=('spend', 'sum'), revenue=('revenue', 'sum')
            ).reset_index()
            sub   = df_wkly[df_wkly['campaign_type'].str.upper() == c_type]
            color = RC_COLORS.get(c_type, '#374151')

            fig_rc = go.Figure()
            if not sub.empty:
                fig_rc.add_trace(go.Scatter(
                    x=sub['spend'], y=sub['revenue'],
                    mode='markers', name='Historical',
                    marker=dict(color=color, size=5, opacity=0.5),
                ))
            fig_rc.add_trace(go.Scatter(
                x=x_range, y=y_merged, mode='lines', name='Response Curve',
                line=dict(color=color, width=2),
            ))
            fig_rc.add_vrect(x0=0, x1=max_obs, fillcolor=color, opacity=0.04,
                layer="below", line_width=0,
                annotation_text="Observed", annotation_font_size=9,
                annotation_font_color=color, annotation_position="top left")
            fig_rc.add_vline(x=max_obs, line_dash="dot", line_color=color, line_width=1, opacity=0.5)
            fig_rc.update_layout(
                title=f"{c_type.title()} ({'Fitted' if fit_ok else 'Fallback'})",
                xaxis_title="Spend ($)", yaxis_title="Revenue ($)",
                legend=dict(orientation='h', yanchor='bottom', y=1.01, x=0),
                **CHART_THEME, height=320,
            )
            _apply_light_axes(fig_rc)
            st.plotly_chart(fig_rc, use_container_width=True)

            if max_obs > 0:
                y_at_max    = float(slope * max_obs)
                y_at_2x     = float(y_merged[np.argmin(np.abs(x_range - max_obs * 2))])
                diminish_pct = ((y_at_2x / y_at_max) - 1) * 100 if y_at_max > 0 else 0
                st.markdown(f"""
                <div class="card-sm" style="font-size:12px;color:#6B7280;margin-top:-4px;">
                  <b style="color:{color}">Max observed:</b> {fmt_dollar(max_obs)} &nbsp;·&nbsp;
                  <b style="color:{color}">Rev at 2× spend:</b> +{diminish_pct:.0f}% &nbsp;·&nbsp;
                  <b style="color:{color}">Hill n:</b> {n_hill:.2f}
                  {'&nbsp;&nbsp;<span style="color:#059669;font-weight:600;">✓ Fitted</span>' if fit_ok else '&nbsp;&nbsp;<span style="color:#D97706;font-weight:600;">⚡ Fallback</span>'}
                </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Normalised Response Overlay</div>', unsafe_allow_html=True)
    fig_overlay = go.Figure()
    for c_type in campaign_types:
        if c_type not in curves: continue
        p        = curves[c_type]
        max_obs  = p['max_observed_spend']
        slope    = p['slope']
        Vmax     = p['Vmax']
        K        = p['K']
        n_hill   = p['n']
        x_norm   = np.linspace(0, 2.5, 200)
        x_actual = x_norm * max_obs
        y_arr = np.array([
            slope * xi if xi <= max_obs else
            (slope * max_obs * hill_function(xi, Vmax, K, n_hill) /
             hill_function(max_obs, Vmax, K, n_hill)
             if hill_function(max_obs, Vmax, K, n_hill) > 0 else slope * xi)
            for xi in x_actual
        ])
        y_norm = y_arr / (slope * max_obs) if (slope * max_obs) > 0 else y_arr
        fig_overlay.add_trace(go.Scatter(
            x=x_norm, y=y_norm, mode='lines', name=c_type.title(),
            line=dict(color=RC_COLORS.get(c_type, '#374151'), width=2),
        ))
    fig_overlay.add_vline(x=1.0, line_dash="dot", line_color="#9CA3AF",
                          annotation_text="Max Observed", annotation_font_size=9)
    fig_overlay.update_layout(
        xaxis_title="Spend (× max observed)", yaxis_title="Revenue (normalised)",
        title="Spend Efficiency by Campaign Type",
        **CHART_THEME, height=300,
    )
    _apply_light_axes(fig_overlay)
    st.plotly_chart(fig_overlay, use_container_width=True)
    st.caption("A flatter slope beyond 1.0× indicates more rapid diminishing returns at the margin.")

# ═════════════════════════════════════════════════════════════
# PAGE: DATA INGESTION
# ═════════════════════════════════════════════════════════════
elif page == "Data Ingestion":
    st.markdown('<div class="section-title">Campaign Data Ingestion</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload Google Ads, Meta Ads, and Bing Ads CSV campaign data to refresh the intelligence pipeline.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="upload-zone uz-google"><div class="uz-title">Google Ads</div></div>', unsafe_allow_html=True)
        google_file = st.file_uploader("google_ads_campaign_stats.csv", type=['csv'], key="gf")
        if google_file: st.success("Google Ads file staged.")
    with col2:
        st.markdown('<div class="upload-zone uz-meta"><div class="uz-title">Meta Ads</div></div>', unsafe_allow_html=True)
        meta_file = st.file_uploader("meta_ads_campaign_stats.csv", type=['csv'], key="mf")
        if meta_file: st.success("Meta Ads file staged.")
    with col3:
        st.markdown('<div class="upload-zone uz-bing"><div class="uz-title">Bing Ads</div></div>', unsafe_allow_html=True)
        bing_file = st.file_uploader("bing_campaign_stats.csv", type=['csv'], key="bf")
        if bing_file: st.success("Bing Ads file staged.")

    if st.button("Process & Ingest Files", type="primary"):
        os.makedirs(DATA_DIR, exist_ok=True)
        saved = 0
        for fname, fobj in [
            ("google_ads_campaign_stats.csv", google_file),
            ("meta_ads_campaign_stats.csv",   meta_file),
            ("bing_campaign_stats.csv",        bing_file),
        ]:
            if fobj:
                with open(os.path.join(DATA_DIR, fname), "wb") as f:
                    f.write(fobj.getbuffer())
                saved += 1
        if saved:
            st.success(f"Saved {saved} file(s) to {DATA_DIR}.")
            st.info("Regenerating feature cache and running inference pipeline…")
            df_raw = preprocess_all(DATA_DIR)
            st.write(f"Ingested {df_raw.shape[0]} daily records across {df_raw['campaign_id'].nunique()} campaigns.")
        else:
            st.warning("Please upload at least one file first.")
