"""
SpendScope - Personal Financial Intelligence & Behaviour Analytics Platform.
Built with Streamlit, Plotly, Pandas, Scikit-learn, and SQLite.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    calculate_financial_kpis,
    get_category_breakdown,
    get_monthly_trends,
)
from src.anomaly_detection import ExpenseAnomalyDetector
from src.behavior import build_spending_behavior_profile
from src.budget import DEFAULT_SAMPLE_BUDGETS, evaluate_budgets
from src.categorization import CATEGORY_METADATA, TransactionCategorizer
from src.database import (
    add_custom_rule,
    delete_category_budget,
    delete_custom_rule,
    delete_session,
    get_category_budgets,
    get_custom_rules,
    list_sessions,
    load_session,
    save_session,
    set_category_budget,
)
from src.forecasting import generate_financial_outlook
from src.ingestion import load_and_validate_csv
from src.insights import generate_spending_insights
from src.preprocessing import format_currency, preprocess_transactions
from src.recurring import (
    detect_recurring_expenses,
    identify_subscription_leakage,
)
from src.shifts import analyze_spending_shifts
from src.layman_intelligence import (
    generate_layman_report,
    generate_executive_html_briefing,
)


# --- Page Configuration ---
st.set_page_config(
    page_title="SpendScope | Financial Intelligence & Behaviour Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Executive Dark Navy & Purple SaaS Aesthetic ---
CUSTOM_CSS = """
<style>
    :root {
        --bg-main: #0B0F19;
        --bg-surface: #111827;
        --bg-surface-elevated: #161F33;
        --primary: #6366F1;
        --primary-glow: rgba(99, 102, 241, 0.20);
        --accent-purple: #8B5CF6;
        --text-primary: #F9FAFB;
        --text-secondary: #9CA3AF;
        --text-muted: #6B7280;
        --border-subtle: #1F2937;
        --border-accent: rgba(139, 92, 246, 0.22);
        --success: #10B981;
        --danger: #EF4444;
        --warning: #F59E0B;
    }

    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    }

    /* Executive Surface Cards */
    .intel-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-accent);
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
    }

    .metric-slab {
        background: linear-gradient(180deg, #131B2E 0%, #111827 100%);
        border: 1px solid var(--border-accent);
        border-radius: 10px;
        padding: 16px 20px;
    }
    .metric-slab-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
        margin-bottom: 4px;
    }
    .metric-slab-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
    }
    .metric-slab-caption {
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin-top: 4px;
    }

    /* Behavioral Trait Cards */
    .trait-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .trait-badge-high {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .trait-badge-mod {
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .trait-badge-neutral {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Narrative Quotes */
    .narrative-box {
        background: #12192B;
        border-left: 4px solid var(--primary);
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0 18px 0;
        font-size: 0.93rem;
        line-height: 1.55;
        color: #E2E8F0;
    }

    /* Insight Cards */
    .insight-card {
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border: 1px solid var(--border-accent);
    }
    .insight-success { border-left: 4px solid var(--success); background: rgba(16,185,129,0.07); }
    .insight-warning { border-left: 4px solid var(--warning); background: rgba(245,158,11,0.07); }
    .insight-danger  { border-left: 4px solid var(--danger);  background: rgba(239,68,68,0.07); }
    .insight-info    { border-left: 4px solid var(--primary);  background: rgba(99,102,241,0.07); }

    /* Sidebar Navigation Enhancement */
    [data-testid="stSidebar"] {
        background-color: #0D1322;
        border-right: 1px solid var(--border-subtle);
    }

    /* Section header typography */
    h2, h3 { letter-spacing: -0.01em; }

    /* Streamlit metric cards overrides */
    [data-testid="stMetric"] {
        background: var(--bg-surface);
        border: 1px solid var(--border-accent);
        border-radius: 10px;
        padding: 14px 18px;
    }
    [data-testid="stMetricValue"] { color: #FFFFFF; }
    [data-testid="stMetricLabel"] {
        color: var(--text-muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- Cached Intelligence Pipeline ---
@st.cache_data(show_spinner=False)
def run_full_intelligence_pipeline(
    file_bytes: bytes,
    file_name: str,
    contamination: float = 0.05,
    override_currency: Optional[str] = None,
):
    """Execute complete end-to-end analytics and behavioural pipeline."""
    stream = io.BytesIO(file_bytes)
    raw_df, val_result = load_and_validate_csv(stream, file_name=file_name)
    if not val_result.is_valid or raw_df is None:
        return None, val_result, None, None, None, None

    clean_df, prep_report = preprocess_transactions(
        raw_df,
        val_result.detected_columns,
        override_currency=override_currency,
    )
    categorizer = TransactionCategorizer.from_database()
    cat_df = categorizer.categorize_dataframe(clean_df)

    cur_sym = prep_report.currency_symbol
    detector = ExpenseAnomalyDetector(contamination=contamination, random_state=42)
    enriched_df, anomaly_summary = detector.fit_predict(cat_df, currency_symbol=cur_sym)

    recurring_items, recurring_summary = detect_recurring_expenses(enriched_df)
    return enriched_df, val_result, prep_report, anomaly_summary, recurring_items, recurring_summary


# --- Sidebar Navigation & Controls ---
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
            <div style="font-size: 2.1rem; line-height: 1;">💳</div>
            <div>
                <h2 style="margin: 0; padding: 0; font-size: 1.35rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">SpendScope</h2>
                <span style="font-size: 0.72rem; background: #312E81; color: #C7D2FE; padding: 2px 8px; border-radius: 10px; font-weight: 600;">Financial Intelligence Platform</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Platform Modules",
        [
            "1. Financial Intelligence Overview",
            "2. Spending Behaviour Profile",
            "3. Major Spending Shifts",
            "4. Insights Worth Noticing",
            "5. Unusual Transaction Alerts",
            "6. Financial Outlook & Recurring",
            "7. Transaction Explorer",
            "8. Budgets & Goals",
            "9. Custom Rules",
            "10. Saved Sessions",
        ],
        index=0,
    )

    st.markdown("---")

    # Session State indicator if an archived session is active
    if "active_loaded_df" in st.session_state and st.session_state["active_loaded_df"] is not None:
        st.markdown(
            f"""
            <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid #6366F1; padding: 10px 14px; border-radius: 8px; margin-bottom: 12px;">
                <div style="font-size: 0.72rem; color: #A5B4FC; font-weight: 700; text-transform: uppercase;">Active Restored Session</div>
                <div style="font-size: 0.88rem; font-weight: 600; color: #FFFFFF; margin-top: 2px;">{st.session_state.get('active_session_name', 'Archived Session')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔄 Return to Live / Sample Data", use_container_width=True):
            del st.session_state["active_loaded_df"]
            if "active_session_name" in st.session_state:
                del st.session_state["active_session_name"]
            st.rerun()

    st.subheader("Display & Mode")
    layman_mode = st.toggle(
        "✨ Smart Summaries",
        value=True,
        help="Shows plain-English context cards alongside charts — what the numbers mean for you, not just what they are.",
    )

    st.subheader("Data & Preferences")
    data_source = st.radio(
        "Data Source",
        ["Use Default Sample Data", "Upload Statement (CSV / PDF)"],
        index=0,
        disabled=("active_loaded_df" in st.session_state and st.session_state["active_loaded_df"] is not None),
    )

    uploaded_file = None
    if data_source == "Upload Statement (CSV / PDF)":
        uploaded_file = st.file_uploader("Upload bank statement (CSV, TXT, or PDF)", type=["csv", "txt", "pdf"])

    currency_pref = st.selectbox(
        "Display Currency",
        [
            "Auto-Detect from File",
            "$ (USD / CAD / AUD)",
            "€ (EUR)",
            "£ (GBP)",
            "₹ (INR)",
            "¥ (JPY / CNY)",
            "CHF (Swiss Franc)",
            "kr (SEK / NOK / DKK)",
            "R$ (BRL)",
            "zł (PLN)",
            "AED",
            "SAR",
        ],
        index=0,
    )

    anomaly_sensitivity = st.slider(
        "Anomaly Contamination Rate",
        min_value=0.01,
        max_value=0.15,
        value=0.05,
        step=0.01,
        help="Proportion of outlier transactions evaluated by Isolation Forest.",
    )


# --- Currency Resolution ---
currency_override_map = {
    "$ (USD / CAD / AUD)": "$",
    "€ (EUR)": "€",
    "£ (GBP)": "£",
    "₹ (INR)": "₹",
    "¥ (JPY / CNY)": "¥",
    "CHF (Swiss Franc)": "CHF",
    "kr (SEK / NOK / DKK)": "kr",
    "R$ (BRL)": "R$",
    "zł (PLN)": "zł",
    "AED": "AED",
    "SAR": "SAR",
}
forced_currency = currency_override_map.get(currency_pref, None)


# --- Data Loading Resolution ---
active_filename = "sample_transactions.csv"
is_restored_session = False

if "active_loaded_df" in st.session_state and st.session_state["active_loaded_df"] is not None:
    df = st.session_state["active_loaded_df"].copy()
    active_filename = st.session_state.get("active_session_name", "Archived Snapshot")
    is_restored_session = True
    val_res = None
    prep_rep = None

    cur_sym = forced_currency or (df["currency_symbol"].iloc[0] if "currency_symbol" in df.columns else "$")
    detector = ExpenseAnomalyDetector(contamination=anomaly_sensitivity, random_state=42)
    df, anom_sum = detector.fit_predict(df, currency_symbol=cur_sym)
    rec_items, rec_sum = detect_recurring_expenses(df)
else:
    file_bytes: Optional[bytes] = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        active_filename = uploaded_file.name
    else:
        sample_file_path = Path("data/sample_transactions.csv")
        if sample_file_path.exists():
            with open(sample_file_path, "rb") as f:
                file_bytes = f.read()
                active_filename = "sample_transactions.csv"

    if file_bytes is None:
        st.warning("Please upload a CSV or PDF file or ensure `data/sample_transactions.csv` exists.")
        st.stop()

    with st.spinner("Processing financial records..."):
        df, val_res, prep_rep, anom_sum, rec_items, rec_sum = run_full_intelligence_pipeline(
            file_bytes,
            active_filename,
            contamination=anomaly_sensitivity,
            override_currency=forced_currency,
        )

    if df is None or (val_res and not val_res.is_valid):
        st.error(f"Failed to ingest dataset: {'; '.join(val_res.errors) if val_res else 'Unknown error'}")
        st.stop()


# Resolve Active Currency Symbol
cur_sym = forced_currency or (df["currency_symbol"].iloc[0] if "currency_symbol" in df.columns and not df.empty else "$")

def fmt(amt: float) -> str:
    """Format monetary amount using resolved currency symbol."""
    return format_currency(amt, cur_sym)


# Precompute Global Financial Metrics & Smart Intelligence Report
kpis = calculate_financial_kpis(df)
behavior_profile = build_spending_behavior_profile(df)
shifts_obj = analyze_spending_shifts(df, currency_symbol=cur_sym)
leakage_obj = identify_subscription_leakage(rec_items, df, currency_symbol=cur_sym)
outlook_obj = generate_financial_outlook(df, recurring_items=rec_items, currency_symbol=cur_sym)

layman_rep = generate_layman_report(
    df=df,
    kpis=kpis,
    behavior_profile=behavior_profile,
    shifts=shifts_obj,
    anom_summary=anom_sum,
    recurring_items=rec_items,
    recurring_leakage=leakage_obj,
    outlook=outlook_obj,
    currency_symbol=cur_sym,
)

min_date = df["date"].min().strftime("%b %d, %Y") if not df.empty else "N/A"
max_date = df["date"].max().strftime("%b %d, %Y") if not df.empty else "N/A"
date_range_str = f"{min_date} – {max_date}"

html_briefing = generate_executive_html_briefing(
    report=layman_rep,
    statement_name=active_filename,
    date_range_str=date_range_str,
    kpis=kpis,
    currency_symbol=cur_sym,
)


# --- Top Header & Persistence Bar ---
col_head_left, col_head_right = st.columns([2.7, 1.7])
with col_head_left:
    module_title = page.split(". ", 1)[1] if ". " in page else page
    st.title(module_title)
    st.caption(f"Statement: **{active_filename}** • {len(df)} transactions ({date_range_str}) • Currency: **{cur_sym}**")

with col_head_right:
    st.write("")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.download_button(
            label="📥 Download Briefing (PDF)",
            data=html_briefing,
            file_name=f"SpendScope_Briefing_{active_filename}.html",
            mime="text/html",
            use_container_width=True,
            help="Download a print-ready HTML report. Open in browser and use Ctrl+P / Cmd+P to save as PDF.",
        )
    with btn_col2:
        if st.button("💾 Archive Snapshot", use_container_width=True, help="Preserve current analysis state in local SQLite database."):
            session_id = save_session(
                df=df,
                session_name=f"Intelligence of {active_filename}",
                file_name=active_filename,
                metadata={"sensitivity": anomaly_sensitivity, "currency": cur_sym},
            )
            st.success(f"Saved! ID: `{session_id}`")


# ==============================================================================
# SECTION 1: FINANCIAL INTELLIGENCE OVERVIEW
# ==============================================================================
if "Overview" in page:
    kpis = calculate_financial_kpis(df)
    monthly_df = get_monthly_trends(df)
    behavior_profile = build_spending_behavior_profile(df)
    comp = behavior_profile.composition

    # Smart Summaries: Intuitive Financial Health Scorecard & $100 Breakdown
    if layman_mode:
        sc = layman_rep.health_scorecard
        hd = layman_rep.hundred_dollar_split

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #111827 0%, #1A1F36 100%); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 12px; padding: 22px 26px; margin-bottom: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.3);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                    <div style="display: flex; align-items: center; gap: 20px;">
                        <div style="width: 76px; height: 76px; border-radius: 50%; border: 4px solid {sc.grade_color}; display: flex; align-items: center; justify-content: center; font-size: 1.85rem; font-weight: 800; color: {sc.grade_color}; background: #0B0F19; box-shadow: 0 0 16px {sc.grade_color}33;">
                            {sc.total_score}
                        </div>
                        <div>
                            <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: #9CA3AF; font-weight: 600;">Personal Financial Health Score</div>
                            <div style="font-size: 1.4rem; font-weight: 800; color: #FFFFFF;">
                                Grade: <span style="color: {sc.grade_color};">{sc.grade_label}</span> <span style="font-size: 0.9rem; color: #9CA3AF; font-weight: 500;">({sc.total_score} / 100)</span>
                            </div>
                            <div style="font-size: 0.88rem; color: #E2E8F0; margin-top: 3px; max-width: 620px; line-height: 1.4;">
                                {sc.summary_verdict}
                            </div>
                        </div>
                    </div>
                    <div style="background: rgba(99, 102, 241, 0.15); border: 1px solid #6366F1; padding: 8px 14px; border-radius: 8px; font-size: 0.82rem; color: #C7D2FE;">
                        ✨ <b>Smart Summaries</b> active
                    </div>
                </div>

                <div style="margin-top: 18px; padding-top: 14px; border-top: 1px solid rgba(255, 255, 255, 0.08);">
                    <div style="font-size: 0.88rem; font-weight: 700; color: #FFFFFF; margin-bottom: 4px;">
                        Where Every {cur_sym}100 You Earn Actually Goes
                    </div>
                    <div style="font-size: 0.84rem; color: #CBD5E1; margin-bottom: 10px;">
                        {hd.summary_text}
                    </div>
                    <div style="display: flex; height: 14px; border-radius: 7px; overflow: hidden; background: #1F2937;">
                        <div style="background: #6366F1; width: {max(2.0, hd.fixed_bills)}%;" title="Fixed Bills: {hd.fixed_bills}%"></div>
                        <div style="background: #14B8A6; width: {max(2.0, hd.flexible_essentials)}%;" title="Essentials: {hd.flexible_essentials}%"></div>
                        <div style="background: #EC4899; width: {max(2.0, hd.discretionary_fun)}%;" title="Fun & Dining: {hd.discretionary_fun}%"></div>
                        <div style="background: #10B981; width: {max(2.0, max(0, hd.saved_pocket))}%;" title="Saved: {max(0, hd.saved_pocket)}%"></div>
                    </div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #9CA3AF; margin-top: 6px;">
                        <span>■ Fixed Bills: <b>{cur_sym}{hd.fixed_bills:.1f}</b></span>
                        <span>■ Daily Living: <b>{cur_sym}{hd.flexible_essentials:.1f}</b></span>
                        <span>■ Fun & Dining: <b>{cur_sym}{hd.discretionary_fun:.1f}</b></span>
                        <span>■ Kept / Saved: <b>{cur_sym}{max(0, hd.saved_pocket):.1f}</b></span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 4 Evaluation Pillars
        with st.expander("View the 4 Pillars behind your Health Score"):
            pillar_cols = st.columns(4)
            for idx, p in enumerate(sc.pillars):
                with pillar_cols[idx]:
                    st.markdown(
                        f"""
                        <div style="background: #131B2E; border: 1px solid #1F2937; border-left: 4px solid {p.status_color}; border-radius: 8px; padding: 12px 14px; height: 100%;">
                            <div style="font-size: 0.75rem; text-transform: uppercase; color: #9CA3AF; font-weight: 600;">{p.name}</div>
                            <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin-top: 2px;">
                                {p.score} <span style="font-size: 0.8rem; color: #9CA3AF;">/ {p.max_score} pts</span>
                            </div>
                            <div style="font-size: 0.82rem; font-weight: 600; color: {p.status_color}; margin-top: 4px;">{p.headline}</div>
                            <div style="font-size: 0.78rem; color: #CBD5E1; margin-top: 4px; line-height: 1.4;">{p.layman_explanation}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Top Smart Takeaways
        st.subheader("What This Means For You")
        t_c1, t_c2 = st.columns(2)
        takeaways_to_show = layman_rep.top_takeaways[:4]
        for idx, t in enumerate(takeaways_to_show):
            target_col = t_c1 if idx % 2 == 0 else t_c2
            with target_col:
                st.markdown(
                    f"""
                    <div style="background: #111827; border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
                        <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; color: #FFFFFF; font-size: 0.98rem;">
                            <span>{t.icon}</span> <span>{t.headline}</span>
                        </div>
                        <div style="font-size: 0.86rem; color: #CBD5E1; margin: 6px 0 8px 0; line-height: 1.5;">
                            {t.plain_explanation}
                        </div>
                        <div style="font-size: 0.8rem; color: #38BDF8; background: #161F33; border: 1px solid #1F2937; padding: 6px 10px; border-radius: 6px;">
                            💡 <b>Action Tip:</b> {t.action_tip}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)

    # Executive Metric Slabs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-slab">
                <div class="metric-slab-label">Total Outflow</div>
                <div class="metric-slab-value" style="color: #F87171;">{fmt(kpis.total_expenses)}</div>
                <div class="metric-slab-caption">{kpis.transaction_count} recorded expenses</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        net_color = "#10B981" if kpis.net_savings >= 0 else "#EF4444"
        st.markdown(
            f"""
            <div class="metric-slab">
                <div class="metric-slab-label">Net Savings</div>
                <div class="metric-slab-value" style="color: {net_color};">{fmt(kpis.net_savings)}</div>
                <div class="metric-slab-caption">Savings Rate: {kpis.savings_rate:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-slab">
                <div class="metric-slab-label">Daily Burn Rate</div>
                <div class="metric-slab-value" style="color: #A78BFA;">{fmt(kpis.avg_daily_spend)}</div>
                <div class="metric-slab-caption">Peak single: {fmt(kpis.largest_expense)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-slab">
                <div class="metric-slab-label">Fixed Obligation Ratio</div>
                <div class="metric-slab-value" style="color: #38BDF8;">{comp.fixed_pct:.1f}%</div>
                <div class="metric-slab-caption">{fmt(comp.fixed_total)} structural burden</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Visualizations Row
    col_chart1, col_chart2 = st.columns([1.6, 1.1])

    with col_chart1:
        st.subheader("Cash Flow Trajectory & Net Velocity")
        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Bar(
                x=monthly_df["year_month"],
                y=monthly_df["income"],
                name="Inflow (Earnings)",
                marker_color="#10B981",
                opacity=0.85,
            )
        )
        fig_trend.add_trace(
            go.Bar(
                x=monthly_df["year_month"],
                y=monthly_df["expense"],
                name="Outflow (Expenses)",
                marker_color="#EF4444",
                opacity=0.85,
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=monthly_df["year_month"],
                y=monthly_df["net_savings"],
                name="Net Retained Capital",
                mode="lines+markers",
                line=dict(color="#8B5CF6", width=3),
                marker=dict(size=7),
            )
        )
        fig_trend.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9CA3AF"),
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#1F2937", tickprefix=cur_sym),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_chart2:
        st.subheader("Structural Spend Composition")
        comp_df = pd.DataFrame([
            {"Structural Bucket": "Fixed Commitments", "Amount": comp.fixed_total, "Share": comp.fixed_pct, "Color": "#6366F1"},
            {"Structural Bucket": "Discretionary Choices", "Amount": comp.discretionary_total, "Share": comp.discretionary_pct, "Color": "#EC4899"},
            {"Structural Bucket": "Flexible Essentials", "Amount": comp.flexible_total, "Share": comp.flexible_pct, "Color": "#14B8A6"},
        ])
        fig_donut = px.pie(
            comp_df,
            values="Amount",
            names="Structural Bucket",
            color="Structural Bucket",
            color_discrete_map={row["Structural Bucket"]: row["Color"] for _, row in comp_df.iterrows()},
            hole=0.62,
        )
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9CA3AF"),
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="v", y=0.5),
        )
        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent",
            hovertemplate=f"<b>%{{label}}</b><br>Amount: {cur_sym}%{{value:,.2f}}<br>Proportion: %{{percent}}",
        )
        st.plotly_chart(fig_donut, use_container_width=True)


# ==============================================================================
# SECTION 2: SPENDING BEHAVIOUR PROFILE
# ==============================================================================
elif "Behaviour" in page:
    profile = build_spending_behavior_profile(df)
    ww = profile.weekday_weekend
    mc = profile.merchant_concentration
    dow = profile.day_of_week_distribution

    st.subheader("Evidence-Based Behavioural Diagnostics")
    st.caption("Quantitative patterns derived strictly from empirical transaction timing, merchant distribution, and volatility.")

    if layman_mode:
        st.markdown(
            f"""
            <div style="background: rgba(99, 102, 241, 0.12); border-left: 4px solid #6366F1; border-radius: 0 8px 8px 0; padding: 14px 18px; margin-bottom: 20px;">
                <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 6px;">In plain terms:</div>
                <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.55;">
                    • <b>Your Core Habit:</b> {layman_rep.top_habit_summary}<br>
                    • <b>Weekend Factor:</b> On Saturdays and Sundays, you spend an average of <b>{fmt(ww.avg_daily_weekend)}/day</b>, which is <b>{ww.weekend_intensity_ratio:.1f}x higher</b> than workdays ({fmt(ww.avg_daily_weekday)}/day). Most weekend money went to <b>{ww.top_weekend_category}</b>.<br>
                    • <b>Where You Shop:</b> You spend money at {mc.total_unique_merchants} places, but <b>{mc.top_3_share_pct:.0f}%</b> of your money went to just 3 stores.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Behavioral Profile Archetype Badges
    t_cols = st.columns(len(profile.behavioral_traits))
    for idx, trait in enumerate(profile.behavioral_traits):
        badge_cls = (
            "trait-badge-high" if trait.significance == "High"
            else ("trait-badge-mod" if trait.significance == "Moderate" else "trait-badge-neutral")
        )
        with t_cols[idx]:
            st.markdown(
                f"""
                <div class="intel-card" style="height: 100%;">
                    <div class="trait-badge {badge_cls}">{trait.archetype}</div>
                    <div style="font-weight: 700; font-size: 1.05rem; color: #FFFFFF; margin-bottom: 6px;">{trait.title}</div>
                    <div style="font-size: 0.85rem; color: #9CA3AF; margin-bottom: 10px;">{trait.description}</div>
                    <div style="font-size: 0.82rem; background: #161F33; border: 1px solid #1F2937; padding: 8px 12px; border-radius: 6px; color: #E2E8F0;">
                        <b>Evidence:</b> {trait.evidence}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Weekday vs Weekend Analysis & Day-of-Week Distribution
    c_ww1, c_ww2 = st.columns([1.1, 1.5])

    with c_ww1:
        st.subheader("Weekday vs Weekend Outflow")
        st.markdown(
            f"""
            <div class="intel-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                    <div>
                        <div style="font-size: 0.75rem; color: #9CA3AF; text-transform: uppercase;">Weekday Volume (Mon-Fri)</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #FFFFFF;">{fmt(ww.weekday_total)} <span style="font-size: 0.85rem; color: #9CA3AF;">({ww.weekday_pct:.1f}%)</span></div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Avg/Day: {fmt(ww.avg_daily_weekday)} • Top: {ww.top_weekday_category}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.75rem; color: #9CA3AF; text-transform: uppercase;">Weekend Volume (Sat-Sun)</div>
                        <div style="font-size: 1.35rem; font-weight: 700; color: #F59E0B;">{fmt(ww.weekend_total)} <span style="font-size: 0.85rem; color: #9CA3AF;">({ww.weekend_pct:.1f}%)</span></div>
                        <div style="font-size: 0.8rem; color: #6B7280;">Avg/Day: {fmt(ww.avg_daily_weekend)} • Top: {ww.top_weekend_category}</div>
                    </div>
                </div>
                <div style="background: #1F2937; border-radius: 6px; height: 10px; width: 100%; overflow: hidden; display: flex;">
                    <div style="background: #6366F1; width: {ww.weekday_pct}%;"></div>
                    <div style="background: #F59E0B; width: {ww.weekend_pct}%;"></div>
                </div>
                <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 10px;">
                    <b>Weekend Intensity:</b> {ww.weekend_intensity_ratio:.2f}x daily spend rate of weekdays.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Day of week bar chart
        fig_dow = px.bar(
            dow,
            x="day_name",
            y="amount",
            labels={"day_name": "Day", "amount": "Spend"},
            color="amount",
            color_continuous_scale=["#312E81", "#6366F1", "#A855F7"],
        )
        fig_dow.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9CA3AF"),
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#1F2937", tickprefix=cur_sym),
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    with c_ww2:
        st.subheader("Merchant Concentration & Vendor Centrality")
        st.markdown(
            f"""
            <div class="intel-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="font-size: 0.75rem; color: #9CA3AF; text-transform: uppercase;">Concentration Status</span>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF;">{mc.concentration_level} (HHI: {mc.hhi_index:,.0f})</div>
                    </div>
                    <div style="text-align: right;">
                        <span style="font-size: 0.75rem; color: #9CA3AF; text-transform: uppercase;">Pareto 80/20 Metric</span>
                        <div style="font-size: 1.25rem; font-weight: 700; color: #8B5CF6;">{mc.pareto_80_merchant_count} of {mc.total_unique_merchants} Vendors</div>
                    </div>
                </div>
                <div style="font-size: 0.82rem; color: #9CA3AF;">
                    Top 3 vendors absorb <b>{mc.top_3_share_pct:.1f}%</b> of total outflow; top 5 absorb <b>{mc.top_5_share_pct:.1f}%</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Top merchants horizontal bar chart
        top_m_table = mc.top_merchants_table.head(10).copy()
        fig_merch = px.bar(
            top_m_table,
            x="amount",
            y="merchant",
            orientation="h",
            color="amount",
            color_continuous_scale=["#3730A3", "#6366F1"],
            hover_data=["category", "pct", "count"],
        )
        fig_merch.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9CA3AF"),
            margin=dict(l=0, r=0, t=10, b=0),
            coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending", showgrid=False),
            xaxis=dict(gridcolor="#1F2937", tickprefix=cur_sym),
        )
        st.plotly_chart(fig_merch, use_container_width=True)


# ==============================================================================
# SECTION 3: MAJOR SPENDING SHIFTS
# ==============================================================================
elif "Shifts" in page:
    st.subheader("Month-over-Month Spending Shift Analysis")
    st.caption("Decomposing net expenditure movement into exact category and merchant drivers.")

    available_months = sorted(df["year_month"].dropna().unique(), reverse=True)
    if len(available_months) < 2:
        st.info("At least two distinct monthly periods are required to calculate month-over-month shifts.")
    else:
        sc1, sc2 = st.columns([1.5, 1.5])
        with sc1:
            target_month = st.selectbox("Current Month to Evaluate", options=available_months, index=0)
        with sc2:
            prev_options = [m for m in available_months if m != target_month]
            default_prev_idx = 0 if prev_options else 0
            baseline_month = st.selectbox("Baseline Comparison Month", options=prev_options, index=default_prev_idx)

        shift = analyze_spending_shifts(
            df,
            curr_month=target_month,
            prev_month=baseline_month,
            currency_symbol=cur_sym,
        )

        if shift is not None:
            # Shift Summary Metrics Bar
            sm1, sm2, sm3, sm4 = st.columns(4)
            with sm1:
                st.metric(f"{shift.curr_month} Outflow", fmt(shift.curr_total_expense))
            with sm2:
                st.metric(
                    f"Net Change from {shift.prev_month}",
                    fmt(shift.net_expense_change),
                    f"{shift.pct_expense_change:+.1f}%",
                    delta_color="inverse",
                )
            with sm3:
                st.metric(
                    "Savings Rate Shift",
                    f"{shift.curr_savings_rate:.1f}%",
                    f"{shift.savings_rate_change_pts:+.1f}% pts",
                )
            with sm4:
                st.metric(
                    "Income Movement",
                    fmt(shift.curr_total_income),
                    f"{((shift.curr_total_income - shift.prev_total_income)/max(1, shift.prev_total_income)*100):+.1f}%",
                )

            if layman_mode:
                more_less = "more" if shift.net_expense_change >= 0 else "less"
                top_exp = shift.top_expansion_categories[0].category if shift.top_expansion_categories else "None"
                top_con = shift.top_contraction_categories[0].category if shift.top_contraction_categories else "None"
                top_exp_amt = fmt(shift.top_expansion_categories[0].abs_change) if shift.top_expansion_categories else fmt(0)
                top_con_amt = fmt(abs(shift.top_contraction_categories[0].abs_change)) if shift.top_contraction_categories else fmt(0)

                st.markdown(
                    f"""
                    <div style="background: rgba(99, 102, 241, 0.12); border-left: 4px solid #6366F1; border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 14px 0 18px 0;">
                        <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 4px;">Monthly Change Summary:</div>
                        <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.55;">
                            In <b>{shift.curr_month}</b>, you spent <b>{fmt(abs(shift.net_expense_change))} {more_less}</b> ({shift.pct_expense_change:+.1f}%) compared to {shift.prev_month}.<br>
                            • <b>Main Spending Surge:</b> <b>{top_exp}</b> increased by {top_exp_amt}.<br>
                            • <b>Main Budget Relief:</b> <b>{top_con}</b> decreased by {top_con_amt}.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Natural Language Evidence Box
            st.markdown(
                f"""
                <div class="narrative-box">
                    💡 <b>Analytical Explanation:</b><br>{shift.narrative_explanation}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Waterfall Decomposition Chart
            st.subheader(f"Spending Shift Waterfall: {shift.prev_month} ➔ {shift.curr_month}")
            wf = shift.waterfall_data
            fig_wf = go.Figure(
                go.Waterfall(
                    name="Month Shift",
                    orientation="v",
                    measure=["absolute" if t == "total" else "relative" for t in wf["type"]],
                    x=wf["label"],
                    textposition="outside",
                    text=[fmt(amt) for amt in wf["amount"]],
                    y=wf["amount"],
                    connector=dict(line=dict(color="#4B5563")),
                    decreasing=dict(marker=dict(color="#10B981")),
                    increasing=dict(marker=dict(color="#EF4444")),
                    totals=dict(marker=dict(color="#6366F1")),
                )
            )
            fig_wf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9CA3AF"),
                margin=dict(l=0, r=0, t=20, b=0),
                yaxis=dict(gridcolor="#1F2937", tickprefix=cur_sym),
            )
            st.plotly_chart(fig_wf, use_container_width=True)

            # Drivers Decomposition Columns
            col_exp, col_con = st.columns(2)
            with col_exp:
                st.subheader("Top Category Expansions (Spend Surges)")
                if shift.top_expansion_categories:
                    exp_data = []
                    for c in shift.top_expansion_categories:
                        exp_data.append({
                            "Category": c.category,
                            "Baseline": fmt(c.prev_amount),
                            "Current": fmt(c.curr_amount),
                            "Increase": f"+{fmt(c.abs_change)}",
                            "Change %": f"+{c.pct_change:.1f}%",
                            "Contribution": f"{c.contribution_share_pct:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(exp_data), use_container_width=True)
                else:
                    st.info("No category recorded a spending increase.")

            with col_con:
                st.subheader("Top Category Contractions (Budget Relief)")
                if shift.top_contraction_categories:
                    con_data = []
                    for c in shift.top_contraction_categories:
                        con_data.append({
                            "Category": c.category,
                            "Baseline": fmt(c.prev_amount),
                            "Current": fmt(c.curr_amount),
                            "Reduction": f"-{fmt(abs(c.abs_change))}",
                            "Change %": f"{c.pct_change:.1f}%",
                            "Contribution": f"{c.contribution_share_pct:.1f}%",
                        })
                    st.dataframe(pd.DataFrame(con_data), use_container_width=True)
                else:
                    st.info("No category recorded a spending reduction.")


# ==============================================================================
# SECTION 4: INSIGHTS WORTH NOTICING
# ==============================================================================
elif "Insights" in page:
    st.subheader("Curated Financial Diagnostics")
    st.caption("Targeted analytical flags highlighting structural anomalies, budgetary concentration, and velocity.")

    monthly_df = get_monthly_trends(df)
    insights = generate_spending_insights(df, monthly_df, currency_symbol=cur_sym)

    if not insights:
        st.info("No notable threshold triggers encountered for this transaction period.")
    else:
        for ins in insights:
            badge_class = f"insight-{ins.type}"
            icon = "✓" if ins.type == "success" else ("!" if ins.type == "warning" else ("▲" if ins.type == "danger" else "i"))
            st.markdown(
                f"""
                <div class="insight-card {badge_class}">
                    <div style="font-weight: 700; font-size: 1rem; color: #FFFFFF; display: flex; justify-content: space-between;">
                        <span>[{icon}] {ins.title}</span>
                        <span style="font-size: 0.72rem; color: #9CA3AF; text-transform: uppercase;">Impact: {ins.impact}</span>
                    </div>
                    <div style="color: #D1D5DB; font-size: 0.9rem; margin-top: 6px;">
                        {ins.message}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# SECTION 5: UNUSUAL TRANSACTION ALERTS
# ==============================================================================
elif "Unusual" in page:
    st.subheader("Explainable Anomaly Intelligence")
    st.markdown(
        """
        <div style="background: rgba(99, 102, 241, 0.12); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; font-size: 0.88rem; color: #CBD5E1;">
            <b>Statistical Anomaly Notice:</b> Isolation Forest flags mathematical outliers based on category distributions,
            first-time merchants, and temporal variance. <b>Anomalies represent statistical deviations, NOT confirmed fraud.</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Evaluated", f"{anom_sum.total_evaluated}")
    with m2:
        st.metric("Outliers Detected", f"{anom_sum.anomalies_detected}", f"{anom_sum.anomaly_percentage:.1f}%")
    with m3:
        st.metric("High Severity", f"{anom_sum.high_severity_count}")
    with m4:
        st.metric("Anomalous Outflow", fmt(anom_sum.total_anomalous_spend))

    if layman_mode:
        st.markdown(
            f"""
            <div style="background: rgba(99, 102, 241, 0.12); border-left: 4px solid #6366F1; border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 16px 0;">
                <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 4px;">How to understand these alerts:</div>
                <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.55;">
                    {layman_rep.outlier_plain_summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Anomaly Scatter Plot
    expense_data = df[df["transaction_type"] == "expense"].copy()
    fig_anom = px.scatter(
        expense_data,
        x="date",
        y="amount",
        color="is_anomaly",
        color_discrete_map={True: "#EF4444", False: "#3B82F6"},
        size=expense_data["amount"].clip(lower=10, upper=1200),
        hover_data=["merchant", "category", "anomaly_severity", "anomaly_tag"],
        title="Transaction Volume Timeline with Flagged Statistical Outliers",
    )
    fig_anom.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9CA3AF"),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="#1F2937", tickprefix=cur_sym),
    )
    st.plotly_chart(fig_anom, use_container_width=True)

    # Anomaly Cards with Contextual Explanations
    st.subheader("Flagged Outlier Registry & Baseline Comparisons")
    anomalies_only = expense_data[expense_data["is_anomaly"]].sort_values(by="amount", ascending=False)

    if anomalies_only.empty:
        st.success("No statistical outliers detected under current sensitivity threshold.")
    else:
        for _, row in anomalies_only.iterrows():
            sev_color = "#EF4444" if row["anomaly_severity"] == "High" else "#F59E0B"
            tag = row.get("anomaly_tag", "Statistical Outlier")
            st.markdown(
                f"""
                <div style="background: #111827; border: 1px solid var(--border-accent); border-left: 5px solid {sev_color}; padding: 16px 20px; border-radius: 8px; margin-bottom: 12px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 700; font-size: 1.15rem; color: #FFFFFF;">{row['merchant']} — {fmt(row['amount'])}</span>
                        <div>
                            <span style="background: #1E1B4B; color: #C7D2FE; padding: 3px 10px; border-radius: 12px; font-weight: 600; font-size: 0.75rem; margin-right: 6px;">{tag}</span>
                            <span style="background: rgba(239, 68, 68, 0.18); color: {sev_color}; padding: 3px 10px; border-radius: 12px; font-weight: 600; font-size: 0.75rem;">{row['anomaly_severity']} Severity</span>
                        </div>
                    </div>
                    <div style="color: #9CA3AF; font-size: 0.85rem; margin-top: 4px;">
                        Date: {row['date'].strftime('%b %d, %Y')} • Category: <b>{row['category']}</b> • Raw Record: <i>{row['description']}</i>
                    </div>
                    <div style="color: #E2E8F0; margin-top: 10px; font-size: 0.92rem; background: #161F33; border: 1px solid #1F2937; padding: 10px 14px; border-radius: 6px;">
                        <b>Contextual Explanation:</b> {row['anomaly_explanation']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ==============================================================================
# SECTION 6: FINANCIAL OUTLOOK & RECURRING COMMITMENTS
# ==============================================================================
elif "Outlook" in page:
    st.subheader("Baseline Spending Forecast & Subscription Leakage")
    st.caption("Historical run-rate projections and recurring obligation audit.")

    if layman_mode:
        st.markdown(
            f"""
            <div style="background: rgba(99, 102, 241, 0.12); border-left: 4px solid #6366F1; border-radius: 0 8px 8px 0; padding: 14px 18px; margin-bottom: 20px;">
                <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem; margin-bottom: 4px;">What to plan for next month:</div>
                <div style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.55;">
                    • <b>Projected Spending:</b> {layman_rep.outlook_plain_summary}<br>
                    • <b>Automatic Bills & Subscriptions:</b> {layman_rep.subscription_plain_summary}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    outlook = generate_financial_outlook(df, recurring_items=rec_items, currency_symbol=cur_sym)
    leakage = identify_subscription_leakage(rec_items, df, currency_symbol=cur_sym)

    if outlook is not None:
        # Forecast Projection Slabs
        fo1, fo2, fo3, fo4 = st.columns(4)
        with fo1:
            st.metric(f"Projected Outflow ({outlook.forecast_month})", fmt(outlook.expected_total_spend))
        with fo2:
            st.metric(
                "Variance Projection Band",
                f"{fmt(outlook.conservative_spend_lower)} – {fmt(outlook.aggressive_spend_upper)}",
                f"±1.0σ",
            )
        with fo3:
            st.metric("Fixed Commitment Floor", fmt(outlook.expected_fixed_commitments))
        with fo4:
            st.metric("Forecast Confidence", f"{outlook.confidence_score * 100:.0f}%", f"{outlook.history_months_count} mos baseline")

        # Fan Chart
        st.subheader("Historical Burn Rate & Next-Month Expectation Band")
        fig_fore = go.Figure()
        
        hist_pts = outlook.timeline_df[outlook.timeline_df["type"] == "Historical Actual"]
        proj_pts = outlook.timeline_df[outlook.timeline_df["type"] == "Projected Forecast"]

        fig_fore.add_trace(
            go.Scatter(
                x=hist_pts["month"],
                y=hist_pts["spend"],
                mode="lines+markers",
                name="Historical Actual",
                line=dict(color="#6366F1", width=3),
                marker=dict(size=7),
            )
        )
        # Connect last historical to projection
        combined_x = [hist_pts["month"].iloc[-1], proj_pts["month"].iloc[0]]
        combined_y = [hist_pts["spend"].iloc[-1], proj_pts["spend"].iloc[0]]
        fig_fore.add_trace(
            go.Scatter(
                x=combined_x,
                y=combined_y,
                mode="lines+markers",
                name="Projection Path",
                line=dict(color="#A855F7", width=3, dash="dash"),
                marker=dict(size=9, symbol="diamond"),
            )
        )
        # Upper & Lower bounds
        fig_fore.add_trace(
            go.Scatter(
                x=[proj_pts["month"].iloc[0], proj_pts["month"].iloc[0]],
                y=[proj_pts["lower_bound"].iloc[0], proj_pts["upper_bound"].iloc[0]],
                mode="markers",
                name="Expected Range (±1σ)",
                marker=dict(color="#F59E0B", size=10, symbol="line-ns", line_width=3),
            )
        )

        fig_fore.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9CA3AF"),
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(gridcolor="#1F2937", tickprefix=cur_sym),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig_fore, use_container_width=True)

        # Methodology & Limitations Box
        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid var(--border-subtle); border-radius: 8px; padding: 14px 18px; margin-bottom: 24px;">
                <div style="font-weight: 700; color: #FFFFFF; font-size: 0.9rem; margin-bottom: 4px;">Analytical Disclosures & Forecasting Limitations</div>
                <div style="font-size: 0.82rem; color: #9CA3AF; margin-bottom: 6px;">{outlook.methodology_text}</div>
                <div style="font-size: 0.82rem; color: #EF4444;">{outlook.limitations_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Subscription Leakage Analysis
    st.subheader("Subscription Leakage & Commitment Audit")
    
    sl1, sl2, sl3 = st.columns(3)
    with sl1:
        st.metric("Recurring Items", f"{leakage.summary.total_recurring_items}")
    with sl2:
        st.metric("Monthly Recurring Obligation", fmt(leakage.summary.monthly_recurring_spend))
    with sl3:
        st.metric("Annualized Subscription Burden", fmt(leakage.summary.annual_recurring_spend))

    # Redundancies Detected
    if leakage.redundancies:
        st.subheader("Identified Service Redundancies")
        for red in leakage.redundancies:
            st.markdown(
                f"""
                <div class="insight-card insight-warning">
                    <div style="font-weight: 700; color: #FFFFFF; font-size: 0.98rem; display: flex; justify-content: space-between;">
                        <span>Overlap: {red.domain} ({red.count} Active Services)</span>
                        <span>{fmt(red.combined_annual_cost)}/yr</span>
                    </div>
                    <div style="color: #D1D5DB; font-size: 0.88rem; margin-top: 6px;">
                        {red.insight_note}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Recurring Table
    if rec_items:
        st.subheader("Active Recurring Commitments Schedule")
        rec_table = []
        for it in rec_items:
            rec_table.append({
                "Merchant": it.merchant,
                "Category": it.category,
                "Cadence": it.cadence,
                "Amount": fmt(it.expected_amount),
                "Annual Cost": fmt(it.annual_cost),
                "Confidence": f"{it.confidence * 100:.0f}%",
                "Last Recorded": it.last_date,
                "Next Projected": it.next_projected_date,
            })
        st.dataframe(pd.DataFrame(rec_table), use_container_width=True)


# ==============================================================================
# SECTION 7: TRANSACTION EXPLORER
# ==============================================================================
elif "Transaction" in page:
    st.subheader("Transaction Intelligence Registry")

    fc1, fc2, fc3, fc4 = st.columns([1.5, 1.5, 1, 1.5])
    with fc1:
        search_query = st.text_input("Search Description or Merchant", placeholder="e.g. Starbucks, Uber...")
    with fc2:
        all_categories = sorted(df["category"].dropna().unique())
        selected_cats = st.multiselect("Category Filter", options=all_categories, default=None)
    with fc3:
        type_filter = st.selectbox("Transaction Type", ["All", "Expense", "Income"])
    with fc4:
        anomaly_filter = st.selectbox("Outlier Status", ["All Transactions", "Anomalies Only", "Normal Only"])

    filtered_df = df.copy()
    if search_query:
        q = search_query.lower()
        filtered_df = filtered_df[
            filtered_df["description"].str.lower().str.contains(q)
            | filtered_df["merchant"].str.lower().str.contains(q)
        ]
    if selected_cats:
        filtered_df = filtered_df[filtered_df["category"].isin(selected_cats)]
    if type_filter != "All":
        filtered_df = filtered_df[filtered_df["transaction_type"] == type_filter.lower()]
    if anomaly_filter == "Anomalies Only":
        filtered_df = filtered_df[filtered_df["is_anomaly"]]
    elif anomaly_filter == "Normal Only":
        filtered_df = filtered_df[~filtered_df["is_anomaly"]]

    st.write(f"Displaying **{len(filtered_df)}** transactions (Cumulative Volume: **{fmt(filtered_df['amount'].sum())}**)")

    display_cols = [
        "date",
        "merchant",
        "category",
        "amount",
        "transaction_type",
        "is_anomaly",
        "anomaly_severity",
        "anomaly_tag",
        "description",
    ]
    formatted_table = filtered_df[display_cols].copy()
    formatted_table["date"] = formatted_table["date"].dt.strftime("%Y-%m-%d")
    formatted_table["amount"] = formatted_table["amount"].apply(fmt)

    st.dataframe(
        formatted_table,
        use_container_width=True,
        column_config={
            "is_anomaly": st.column_config.CheckboxColumn("Outlier?"),
            "anomaly_severity": st.column_config.TextColumn("Severity"),
            "anomaly_tag": st.column_config.TextColumn("Diagnostic Tag"),
            "amount": st.column_config.TextColumn("Amount"),
        },
    )

    csv_buf = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Cleaned / Enriched Statement CSV",
        data=csv_buf,
        file_name="spendscope_enriched_transactions.csv",
        mime="text/csv",
    )


# ==============================================================================
# SECTION 8: BUDGETS & GOALS
# ==============================================================================
elif "Budgets" in page:
    st.subheader("Category Budget Targets & Velocity Tracking")

    saved_budgets = get_category_budgets()
    effective_budgets = saved_budgets if saved_budgets else DEFAULT_SAMPLE_BUDGETS

    available_months = sorted(df["year_month"].dropna().unique(), reverse=True)
    b_col1, _ = st.columns([1.5, 2.5])
    with b_col1:
        selected_month = st.selectbox("Select Target Billing Month", options=available_months, index=0)

    evaluations = evaluate_budgets(df, target_year_month=selected_month, budget_limits=effective_budgets)

    total_budget = sum(e.monthly_limit for e in evaluations)
    total_actual = sum(e.current_spend for e in evaluations)
    total_rem = total_budget - total_actual
    overall_pct = (total_actual / total_budget * 100.0) if total_budget > 0 else 0.0

    bk1, bk2, bk3, bk4 = st.columns(4)
    with bk1:
        st.metric("Total Monthly Budget", fmt(total_budget))
    with bk2:
        st.metric("Total Spent", fmt(total_actual))
    with bk3:
        st.metric("Net Buffer Remaining", fmt(total_rem))
    with bk4:
        st.metric("Overall Utilization", f"{overall_pct:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    for item in evaluations:
        status_badge_bg = (
            "rgba(239, 68, 68, 0.2)" if item.status == "Exceeded"
            else ("rgba(245, 158, 11, 0.2)" if item.status == "Near Limit" else "rgba(16, 185, 129, 0.2)")
        )
        bar_fill_width = min(100.0, item.percentage_used)

        st.markdown(
            f"""
            <div style="background: #111827; border: 1px solid var(--border-accent); border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div>
                        <span style="font-weight: 700; color: #FFFFFF; font-size: 1rem;">{item.category}</span>
                        <span style="margin-left: 10px; font-size: 0.78rem; background: {status_badge_bg}; color: {item.status_color}; padding: 2px 8px; border-radius: 10px; font-weight: 600;">{item.status} ({item.percentage_used:.1f}%)</span>
                    </div>
                    <div style="font-weight: 600; font-size: 0.95rem; color: #E2E8F0;">
                        {fmt(item.current_spend)} <span style="color: #9CA3AF; font-size: 0.82rem;">/ {fmt(item.monthly_limit)}</span>
                    </div>
                </div>
                <div style="background: #1F2937; border-radius: 6px; height: 9px; width: 100%; overflow: hidden;">
                    <div style="background: {item.status_color}; height: 100%; width: {bar_fill_width}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #9CA3AF; margin-top: 6px;">
                    <span>Remaining Buffer: <b>{fmt(item.remaining)}</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("⚙️ Adjust Monthly Category Caps"):
        with st.form("budget_form"):
            eb_cat = st.selectbox("Category", options=list(CATEGORY_METADATA.keys()))
            current_cap = effective_budgets.get(eb_cat, 300.0)
            eb_limit = st.number_input(f"Monthly Cap ({cur_sym})", min_value=10.0, max_value=50000.0, value=float(current_cap), step=25.0)
            save_b_btn = st.form_submit_button("Update Budget Cap")
            if save_b_btn:
                set_category_budget(eb_cat, eb_limit)
                st.success(f"Updated {eb_cat} budget to {fmt(eb_limit)}/month!")
                st.rerun()


# ==============================================================================
# SECTION 9: CUSTOM RULES
# ==============================================================================
elif "Rules" in page:
    st.subheader("Custom Category Keyword & Pattern Rules")
    st.caption("Override categorization logic during ingestion using merchant patterns.")

    with st.form("add_rule_form"):
        col_r1, col_r2, col_r3 = st.columns([1.5, 2, 1])
        with col_r1:
            new_cat = st.selectbox("Target Category", options=list(CATEGORY_METADATA.keys()))
        with col_r2:
            new_pattern = st.text_input("Merchant Keyword / Regex", placeholder="e.g. patagonia, openai, bakery")
        with col_r3:
            st.write("")
            submit_rule = st.form_submit_button("➕ Add Rule", use_container_width=True)

        if submit_rule:
            if not new_pattern.strip():
                st.error("Please enter a valid keyword or pattern.")
            else:
                rule_id = add_custom_rule(new_cat, new_pattern)
                st.success(f"Rule registered for '{new_cat}' matching '{new_pattern}'! (Rule ID: {rule_id})")
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    col_r_act1, col_r_act2 = st.columns([2.5, 1.5])
    with col_r_act1:
        st.subheader("Active User-Defined Rules")
    with col_r_act2:
        if st.button("⚡ Apply Rules Retroactively", use_container_width=True, help="Re-categorizes currently loaded transactions with custom rules."):
            cat_engine = TransactionCategorizer.from_database()
            updated_df = cat_engine.categorize_dataframe(df)
            detector = ExpenseAnomalyDetector(contamination=anomaly_sensitivity, random_state=42)
            updated_df, _ = detector.fit_predict(updated_df, currency_symbol=cur_sym)
            st.session_state["active_loaded_df"] = updated_df
            st.success("Successfully applied rules! Active transactions re-categorized.")
            st.rerun()

    user_rules = get_custom_rules()
    if not user_rules:
        st.info("No custom rules configured. SpendScope is using its built-in financial dictionaries.")
    else:
        for r in user_rules:
            r_col1, r_col2, r_col3 = st.columns([3, 1.5, 1])
            with r_col1:
                st.markdown(f"Keyword: `\"{r['pattern']}\"` ➔ <b>{r['category']}</b>", unsafe_allow_html=True)
            with r_col2:
                st.caption(f"Created: {r['created_at'][:10]}")
            with r_col3:
                if st.button("Delete", key=f"del_rule_{r['rule_id']}", use_container_width=True):
                    delete_custom_rule(r["rule_id"])
                    st.success(f"Rule {r['rule_id']} deleted.")
                    st.rerun()


# ==============================================================================
# SECTION 10: SAVED SESSIONS
# ==============================================================================
elif "Sessions" in page:
    st.subheader("SQLite Session Archive & Historical Snapshots")
    st.caption("Review, activate, and manage past financial analysis snapshots stored locally.")

    sessions = list_sessions()
    if not sessions:
        st.info("No analysis sessions saved yet. Click **'Archive Session to SQLite'** in the top-right to preserve the current state.")
    else:
        for s in sessions:
            with st.container():
                st.markdown(
                    f"""
                    <div style="background: #111827; border: 1px solid var(--border-accent); border-radius: 8px; padding: 16px; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 700; font-size: 1.1rem; color: #FFFFFF;">{s['session_name']}</span>
                            <span style="font-size: 0.78rem; background: #1E1B4B; color: #C7D2FE; padding: 3px 10px; border-radius: 12px;">{s['session_id']}</span>
                        </div>
                        <div style="color: #9CA3AF; font-size: 0.85rem; margin-top: 6px;">
                            Source File: <b>{s['file_name']}</b> • Saved: {s['created_at'][:19]} • Records: <b>{s['total_rows']}</b>
                        </div>
                        <div style="color: #E2E8F0; margin-top: 8px; font-size: 0.9rem;">
                            Income: <span style="color: #10B981;">{fmt(s['total_income'])}</span> • Expenses: <span style="color: #F87171;">{fmt(s['total_expense'])}</span> • Net: <b>{fmt(s['net_savings'])}</b>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                s_act_col1, s_act_col2, _ = st.columns([1.5, 1.5, 3])
                with s_act_col1:
                    if st.button("📂 Load & Activate", key=f"load_{s['session_id']}", use_container_width=True):
                        loaded_df = load_session(s["session_id"])
                        if loaded_df is not None:
                            st.session_state["active_loaded_df"] = loaded_df
                            st.session_state["active_session_name"] = s["session_name"]
                            st.success(f"Restored session {s['session_id']}!")
                            st.rerun()
                with s_act_col2:
                    if st.button("🗑️ Delete", key=f"del_{s['session_id']}", use_container_width=True):
                        delete_session(s["session_id"])
                        if st.session_state.get("active_session_name") == s["session_name"]:
                            del st.session_state["active_loaded_df"]
                            del st.session_state["active_session_name"]
                        st.success(f"Session {s['session_id']} deleted.")
                        st.rerun()
