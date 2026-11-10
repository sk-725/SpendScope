"""
SpendScope - Layman Financial Intelligence & Human Translation Engine.

Transforms complex financial indicators, statistical anomaly scores, and time-series
metrics into everyday, plain-English summaries that any user can instantly understand.
Computes an intuitive 0-100 Financial Health Score and generates print-ready executive briefings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.preprocessing import format_currency


@dataclass
class HealthPillarScore:
    """Individual assessment pillar within the 0-100 Financial Health Score."""
    name: str
    score: int          # 0 to 25
    max_score: int = 25
    status: str = "Good"  # Excellent, Good, Fair, Alert
    status_color: str = "#10B981"
    headline: str = ""
    layman_explanation: str = ""


@dataclass
class FinancialHealthScorecard:
    """Composite 0-100 personal financial health evaluation."""
    total_score: int     # 0 to 100
    grade_label: str     # e.g. "Excellent", "Strong", "Fair", "Needs Attention"
    grade_color: str     # Hex color
    summary_verdict: str # Plain English 1-2 sentence verdict
    pillars: List[HealthPillarScore] = field(default_factory=list)


@dataclass
class HundredDollarBreakdown:
    """Human-intuitive breakdown of where every $100 earned actually goes."""
    fixed_bills: float       # e.g. $38.50
    flexible_essentials: float # e.g. $26.20
    discretionary_fun: float # e.g. $19.10
    saved_pocket: float      # e.g. $16.20
    summary_text: str = ""


@dataclass
class LaymanInsightItem:
    """Actionable takeaway presented in conversational English."""
    icon: str
    headline: str
    plain_explanation: str
    action_tip: str
    priority: str = "Normal"  # High, Normal, Info


@dataclass
class LaymanFinancialReport:
    """Unified human-readable financial intelligence pack."""
    health_scorecard: FinancialHealthScorecard
    hundred_dollar_split: HundredDollarBreakdown
    top_takeaways: List[LaymanInsightItem]
    top_habit_summary: str
    outlier_plain_summary: str
    subscription_plain_summary: str
    outlook_plain_summary: str


def compute_financial_health_score(
    df: pd.DataFrame,
    savings_rate: float,
    fixed_burden_pct: float,
    anomalous_spend_pct: float,
    recurring_redundancy_count: int,
    currency_symbol: str = "$",
) -> FinancialHealthScorecard:
    """
    Compute an intuitive 0-100 personal financial health score across 4 pillars (25 pts each).
    
    Pillars:
      1. Savings Habit & Capital Retention (0-25 pts)
      2. Living Essentials Balance vs Fun (0-25 pts)
      3. Outlier & Surprise Stability (0-25 pts)
      4. Subscription & Fixed Leakage Control (0-25 pts)
    """
    # 1. Savings Habit (0-25)
    if savings_rate >= 25.0:
        p1_score = 25
        p1_status = "Excellent"
        p1_color = "#10B981"
        p1_head = f"Top-Tier Savings Rate ({savings_rate:.1f}%)"
        p1_expl = "You are saving more than a quarter of everything you earn. Outstanding wealth accumulation!"
    elif savings_rate >= 15.0:
        p1_score = 21
        p1_status = "Good"
        p1_color = "#34D399"
        p1_head = f"Healthy Savings Cushion ({savings_rate:.1f}%)"
        p1_expl = "You keep a strong portion of your income, comfortably beating the standard 10% benchmark."
    elif savings_rate >= 5.0:
        p1_score = 14
        p1_status = "Fair"
        p1_color = "#F59E0B"
        p1_head = f"Modest Savings Buffer ({savings_rate:.1f}%)"
        p1_expl = "You are in the green, but a small emergency could quickly wipe out monthly savings."
    elif savings_rate >= 0.0:
        p1_score = 8
        p1_status = "Fair"
        p1_color = "#F59E0B"
        p1_head = "Living Paycheck-to-Paycheck (0-5%)"
        p1_expl = "Virtually every dollar coming in is immediately going right back out."
    else:
        p1_score = 2
        p1_status = "Alert"
        p1_color = "#EF4444"
        p1_head = f"Deficit Outflow (Negative Savings)"
        p1_expl = "Expenses are higher than income for this period. You are dipping into savings or credit."

    # 2. Essentials vs Fun Balance (0-25)
    # Target 50/30/20 rule: Fixed commitments around 50% or less
    if fixed_burden_pct <= 45.0:
        p2_score = 25
        p2_status = "Excellent"
        p2_color = "#10B981"
        p2_head = f"Low Fixed Overhead ({fixed_burden_pct:.1f}%)"
        p2_expl = "Non-negotiable bills are low, giving you maximum flexibility to adjust spending whenever needed."
    elif fixed_burden_pct <= 60.0:
        p2_score = 20
        p2_status = "Good"
        p2_color = "#34D399"
        p2_head = f"Balanced Overhead ({fixed_burden_pct:.1f}%)"
        p2_expl = "Your rent and basic bills take up a healthy, manageable portion of total spending."
    elif fixed_burden_pct <= 75.0:
        p2_score = 13
        p2_status = "Fair"
        p2_color = "#F59E0B"
        p2_head = f"High Fixed Burden ({fixed_burden_pct:.1f}%)"
        p2_expl = "A large majority of your money is locked into rigid bills, leaving less room for discretionary fun."
    else:
        p2_score = 6
        p2_status = "Alert"
        p2_color = "#EF4444"
        p2_head = f"Very Heavy Fixed Burden ({fixed_burden_pct:.1f}%)"
        p2_expl = "Over 75% of your money is trapped in bills. Reducing rent or high utility costs is your biggest lever."

    # 3. Outlier & Surprise Stability (0-25)
    if anomalous_spend_pct <= 3.0:
        p3_score = 25
        p3_status = "Excellent"
        p3_color = "#10B981"
        p3_head = "Highly Predictable Spending"
        p3_expl = "Virtually no massive surprise spikes. Your spending routine is disciplined and steady."
    elif anomalous_spend_pct <= 10.0:
        p3_score = 19
        p3_status = "Good"
        p3_color = "#34D399"
        p3_head = "Few Isolated Spikes"
        p3_expl = "A couple of high-ticket purchases occurred, but they remained within reasonable bounds."
    elif anomalous_spend_pct <= 20.0:
        p3_score = 12
        p3_status = "Fair"
        p3_color = "#F59E0B"
        p3_head = f"Noticeable Big-Ticket Spikes ({anomalous_spend_pct:.1f}%)"
        p3_expl = "Unusual large purchases absorbed a substantial chunk of total outflow."
    else:
        p3_score = 5
        p3_status = "Alert"
        p3_color = "#EF4444"
        p3_head = f"High Volatility Spikes ({anomalous_spend_pct:.1f}%)"
        p3_expl = "A large fraction of money went into huge one-off or irregular purchases."

    # 4. Subscription & Redundancy Health (0-25)
    if recurring_redundancy_count == 0:
        p4_score = 25
        p4_status = "Excellent"
        p4_color = "#10B981"
        p4_head = "Zero Duplicate Subscriptions"
        p4_expl = "No overlapping services detected. You aren't wasting cash on duplicate streaming or apps."
    elif recurring_redundancy_count == 1:
        p4_score = 16
        p4_status = "Fair"
        p4_color = "#F59E0B"
        p4_head = "1 Overlapping Subscription Detected"
        p4_expl = "You are paying for multiple services in the same category (e.g. streaming or music) at the same time."
    else:
        p4_score = 8
        p4_status = "Alert"
        p4_color = "#EF4444"
        p4_head = f"{recurring_redundancy_count} Overlapping Subscriptions"
        p4_expl = "Multiple duplicate services detected. Consolidating these gives you instant monthly savings."

    total_score = p1_score + p2_score + p3_score + p4_score
    total_score = max(0, min(100, total_score))

    if total_score >= 85:
        grade_label = "Excellent"
        grade_color = "#10B981"
        summary_verdict = (
            "Your finances are in fantastic shape! You have strong savings habits, low wasteful leakage, "
            "and predictable living costs."
        )
    elif total_score >= 70:
        grade_label = "Good"
        grade_color = "#34D399"
        summary_verdict = (
            "You have solid financial footing with positive savings. A couple of minor adjustments to "
            "subscriptions or weekend outings could elevate you to the top tier."
        )
    elif total_score >= 50:
        grade_label = "Fair"
        grade_color = "#F59E0B"
        summary_verdict = (
            "You are staying afloat, but high overhead or occasional large splurges are eating into your savings. "
            "Focusing on recurring bills and weekend pacing will provide fast relief."
        )
    else:
        grade_label = "Needs Attention"
        grade_color = "#EF4444"
        summary_verdict = (
            "Your outflow is running close to or above your income. Review the flagged outliers and "
            "heavy fixed bills below to reclaim control."
        )

    pillars = [
        HealthPillarScore("Savings Cushion", p1_score, 25, p1_status, p1_color, p1_head, p1_expl),
        HealthPillarScore("Fixed vs Fun Balance", p2_score, 25, p2_status, p2_color, p2_head, p2_expl),
        HealthPillarScore("Spending Stability", p3_score, 25, p3_status, p3_color, p3_head, p3_expl),
        HealthPillarScore("Subscription Efficiency", p4_score, 25, p4_status, p4_color, p4_head, p4_expl),
    ]

    return FinancialHealthScorecard(
        total_score=total_score,
        grade_label=grade_label,
        grade_color=grade_color,
        summary_verdict=summary_verdict,
        pillars=pillars,
    )


def compute_hundred_dollar_breakdown(
    total_income: float,
    total_expense: float,
    fixed_total: float,
    flexible_total: float,
    discretionary_total: float,
    currency_symbol: str = "$",
) -> HundredDollarBreakdown:
    """
    Translates raw totals into a simple $100 paycheck breakdown.
    For every $100 earned, how much went to bills, essentials, fun, and savings?
    """
    if total_income <= 0:
        # Fallback to expense shares if no income recorded
        tot = max(1.0, total_expense)
        f_val = (fixed_total / tot) * 100.0
        x_val = (flexible_total / tot) * 100.0
        d_val = (discretionary_total / tot) * 100.0
        s_val = 0.0
    else:
        f_val = (fixed_total / total_income) * 100.0
        x_val = (flexible_total / total_income) * 100.0
        d_val = (discretionary_total / total_income) * 100.0
        s_val = ((total_income - total_expense) / total_income) * 100.0

    summary = (
        f"For every {currency_symbol}100 that hits your account, roughly "
        f"{currency_symbol}{f_val:.1f} pays non-negotiable bills, "
        f"{currency_symbol}{x_val:.1f} covers groceries & daily living, "
        f"{currency_symbol}{d_val:.1f} goes toward dining & entertainment, "
        f"and {currency_symbol}{max(0, s_val):.1f} stays in your pocket as savings."
    )

    return HundredDollarBreakdown(
        fixed_bills=round(f_val, 1),
        flexible_essentials=round(x_val, 1),
        discretionary_fun=round(d_val, 1),
        saved_pocket=round(s_val, 1),
        summary_text=summary,
    )


def generate_layman_report(
    df: pd.DataFrame,
    kpis: Any,
    behavior_profile: Any,
    shifts: Optional[Any] = None,
    anom_summary: Optional[Any] = None,
    recurring_items: Optional[List[Any]] = None,
    recurring_leakage: Optional[Any] = None,
    outlook: Optional[Any] = None,
    currency_symbol: str = "$",
) -> LaymanFinancialReport:
    """
    Generate an all-in-one layman financial intelligence report with human translations.
    """
    comp = behavior_profile.composition
    ww = behavior_profile.weekday_weekend
    mc = behavior_profile.merchant_concentration

    # Redundancy count
    red_count = len(recurring_leakage.redundancies) if recurring_leakage and recurring_leakage.redundancies else 0

    # Anomalous spend percentage
    anom_pct = anom_summary.anomalous_spend_pct if anom_summary else 0.0

    # 1. Health Scorecard
    scorecard = compute_financial_health_score(
        df=df,
        savings_rate=kpis.savings_rate,
        fixed_burden_pct=comp.fixed_pct,
        anomalous_spend_pct=anom_pct,
        recurring_redundancy_count=red_count,
        currency_symbol=currency_symbol,
    )

    # 2. $100 Breakdown
    hundred_split = compute_hundred_dollar_breakdown(
        total_income=kpis.total_income,
        total_expense=kpis.total_expenses,
        fixed_total=comp.fixed_total,
        flexible_total=comp.flexible_total,
        discretionary_total=comp.discretionary_total,
        currency_symbol=currency_symbol,
    )

    # 3. Top Takeaways in plain English
    takeaways: List[LaymanInsightItem] = []

    # Savings Takeaway
    if kpis.savings_rate >= 15.0:
        takeaways.append(
            LaymanInsightItem(
                icon="💰",
                headline="Strong Savings Momentum",
                plain_explanation=(
                    f"You retained {format_currency(kpis.net_savings, currency_symbol)} ({kpis.savings_rate:.1f}% savings rate). "
                    f"That puts you ahead of most personal finance benchmarks!"
                ),
                action_tip="Automate moving this surplus into a high-yield savings or investment account right on payday.",
                priority="Normal",
            )
        )
    elif kpis.savings_rate > 0:
        takeaways.append(
            LaymanInsightItem(
                icon="⚠️",
                headline="Modest Monthly Buffer",
                plain_explanation=(
                    f"You saved {format_currency(kpis.net_savings, currency_symbol)} ({kpis.savings_rate:.1f}%). "
                    "You're in the green, but one unexpected repair or ticket could wipe this out."
                ),
                action_tip="Try trimming discretionary dining or entertainment by just 10% to double your savings cushion.",
                priority="Normal",
            )
        )
    else:
        takeaways.append(
            LaymanInsightItem(
                icon="🚨",
                headline="Spending Outpaced Income",
                plain_explanation=(
                    f"Your outflow exceeded inflow by {format_currency(abs(kpis.net_savings), currency_symbol)}. "
                    "You are pulling money from past reserves or credit."
                ),
                action_tip="Review the top expense spikes below and pause non-essential purchases this upcoming cycle.",
                priority="High",
            )
        )

    # Weekend vs Weekday Takeaway
    if ww.weekend_intensity_ratio >= 1.5:
        takeaways.append(
            LaymanInsightItem(
                icon="🎉",
                headline=f"Weekend Spend Surge ({ww.weekend_intensity_ratio:.1f}x Daily Rate)",
                plain_explanation=(
                    f"On Saturdays and Sundays, you spend an average of {format_currency(ww.avg_daily_weekend, currency_symbol)} per day, "
                    f"compared to {format_currency(ww.avg_daily_weekday, currency_symbol)} per day during the workweek. "
                    f"Most weekend money went to {ww.top_weekend_category}."
                ),
                action_tip="Setting a dedicated weekend spending cap (e.g. cash envelope or separate card) will quickly prevent budget creep.",
                priority="Normal",
            )
        )

    # Merchant Concentration Takeaway
    if mc.top_3_share_pct >= 40.0:
        top_merch_names = ", ".join(mc.top_merchants_table.head(3)["merchant"].tolist())
        takeaways.append(
            LaymanInsightItem(
                icon="🏪",
                headline="High Spending Concentration at Top Stores",
                plain_explanation=(
                    f"More than {mc.top_3_share_pct:.0f}% of every dollar you spent went to just 3 places: {top_merch_names}."
                ),
                action_tip="Because your spending is so concentrated, negotiating rent, utilities, or shopping grocery sales has an outsized impact.",
                priority="Normal",
            )
        )

    # Subscription Overlap Takeaway
    if recurring_leakage and recurring_leakage.redundancies:
        for red in recurring_leakage.redundancies:
            takeaways.append(
                LaymanInsightItem(
                    icon="📺",
                    headline=f"Duplicate Subscription Alert: {red.domain}",
                    plain_explanation=(
                        f"You are actively paying for {red.count} separate services ({red.domain}) "
                        f"costing a combined {format_currency(red.combined_annual_cost, currency_symbol)} per year."
                    ),
                    action_tip=f"Cancel or pause one of these services to put {format_currency(red.combined_annual_cost / red.count, currency_symbol)}/year back in your pocket.",
                    priority="High",
                )
            )

    # 4. Outlier summary in human terms
    expense_data = df[df["transaction_type"] == "expense"] if "transaction_type" in df.columns else df
    if "is_anomaly" in expense_data.columns:
        anom_rows = expense_data[expense_data["is_anomaly"]].sort_values(by="amount", ascending=False)
    else:
        anom_rows = pd.DataFrame()
    if not anom_rows.empty:
        top_anom = anom_rows.iloc[0]
        outlier_plain = (
            f"We flagged {len(anom_rows)} transactions as unusual. The single biggest surprise was "
            f"**{format_currency(top_anom['amount'], currency_symbol)} at {top_anom['merchant']}** on {top_anom['date'].strftime('%b %d')}. "
            f"This is not fraud, but it was {top_anom.get('anomaly_explanation', 'significantly higher than your normal purchases')}."
        )
    else:
        outlier_plain = "No unusually large or unexpected spending spikes were found. Your spending was remarkably consistent!"

    # 5. Top habit summary
    top_habit = (
        f"Your essential living costs (Housing, Utilities, Groceries) absorb "
        f"**{comp.fixed_pct + comp.flexible_pct:.1f}%** of your total spending, "
        f"leaving **{comp.discretionary_pct:.1f}%** for fun, dining out, and shopping."
    )

    # 6. Subscription summary
    if recurring_items:
        monthly_sub_tot = sum(it.expected_amount for it in recurring_items if it.cadence == "Monthly")
        annual_sub_tot = sum(it.annual_cost for it in recurring_items)
        sub_plain = (
            f"You have **{len(recurring_items)} recurring bills and subscriptions** that cost you roughly "
            f"**{format_currency(monthly_sub_tot, currency_symbol)} every month** ({format_currency(annual_sub_tot, currency_symbol)} per year). "
            f"These are fixed obligations that happen automatically."
        )
    else:
        sub_plain = "No steady recurring monthly subscriptions were detected in this dataset."

    # 7. Outlook summary
    if outlook:
        outlook_plain = (
            f"Looking ahead to {outlook.forecast_month}, we project you will spend roughly "
            f"**{format_currency(outlook.expected_total_spend, currency_symbol)}** "
            f"(likely between {format_currency(outlook.conservative_spend_lower, currency_symbol)} and {format_currency(outlook.aggressive_spend_upper, currency_symbol)}). "
            f"At least {format_currency(outlook.expected_fixed_commitments, currency_symbol)} is already committed to your rent and bills."
        )
    else:
        outlook_plain = "At least two distinct billing cycles are needed to project your next month's spending run-rate."

    return LaymanFinancialReport(
        health_scorecard=scorecard,
        hundred_dollar_split=hundred_split,
        top_takeaways=takeaways,
        top_habit_summary=top_habit,
        outlier_plain_summary=outlier_plain,
        subscription_plain_summary=sub_plain,
        outlook_plain_summary=outlook_plain,
    )


def generate_executive_html_briefing(
    report: LaymanFinancialReport,
    statement_name: str,
    date_range_str: str,
    kpis: Any,
    currency_symbol: str = "$",
) -> str:
    """
    Generate a standalone, beautifully styled, self-contained HTML executive briefing.
    Can be downloaded or opened in browser and printed to PDF with Ctrl+P / Cmd+P.
    """
    sc = report.health_scorecard
    hd = report.hundred_dollar_split

    pillars_html = ""
    for p in sc.pillars:
        pillars_html += f"""
        <div style="background: #1E293B; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; border-left: 4px solid {p.status_color};">
            <div style="display: flex; justify-content: space-between; font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">
                <span>{p.name}</span>
                <span style="color: {p.status_color};">{p.score} / {p.max_score} pts ({p.status})</span>
            </div>
            <div style="font-weight: 600; color: #E2E8F0; font-size: 0.85rem; margin-top: 4px;">{p.headline}</div>
            <div style="color: #94A3B8; font-size: 0.82rem; margin-top: 2px;">{p.layman_explanation}</div>
        </div>
        """

    takeaways_html = ""
    for t in report.top_takeaways:
        takeaways_html += f"""
        <div style="background: #0F172A; border: 1px solid #334155; border-radius: 8px; padding: 14px 18px; margin-bottom: 12px;">
            <div style="font-size: 1rem; font-weight: 700; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                <span>{t.icon}</span> <span>{t.headline}</span>
            </div>
            <div style="color: #CBD5E1; font-size: 0.88rem; margin: 6px 0 8px 0; line-height: 1.5;">
                {t.plain_explanation}
            </div>
            <div style="font-size: 0.82rem; color: #38BDF8; background: #1E293B; padding: 6px 10px; border-radius: 6px;">
                💡 <b>Action Tip:</b> {t.action_tip}
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>SpendScope — Executive Financial Briefing ({statement_name})</title>
    <style>
        @media print {{
            body {{ background: #FFFFFF !important; color: #000000 !important; font-size: 11pt; }}
            .container {{ max-width: 100% !important; padding: 0 !important; }}
            .no-print {{ display: none !important; }}
            .card {{ border: 1px solid #CCCCCC !important; background: #FFFFFF !important; color: #000 !important; box-shadow: none !important; }}
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0B0F19;
            color: #F8FAFC;
            margin: 0;
            padding: 30px 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #334155;
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .brand {{
            font-size: 1.8rem;
            font-weight: 800;
            color: #FFFFFF;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .badge {{
            background: #6366F1;
            color: #FFFFFF;
            font-size: 0.75rem;
            padding: 4px 10px;
            border-radius: 12px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .card {{
            background: #111827;
            border: 1px solid #1F2937;
            border-radius: 10px;
            padding: 20px 24px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        .score-circle {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 80px;
            height: 80px;
            border-radius: 50%;
            border: 4px solid {sc.grade_color};
            font-size: 2rem;
            font-weight: 800;
            color: {sc.grade_color};
        }}
        .stat-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }}
        .stat-box {{
            background: #1E293B;
            border-radius: 8px;
            padding: 12px 16px;
            text-align: center;
        }}
        .stat-val {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #FFFFFF;
        }}
        .stat-lbl {{
            font-size: 0.75rem;
            color: #94A3B8;
            text-transform: uppercase;
            font-weight: 600;
        }}
        .btn {{
            background: #6366F1;
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <div class="brand">💳 SpendScope <span class="badge">Layman Executive Briefing</span></div>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 4px;">Statement: <b>{statement_name}</b> • Period: {date_range_str}</div>
            </div>
            <div class="no-print">
                <button class="btn" onclick="window.print()">🖨️ Print to PDF</button>
            </div>
        </div>

        <!-- KPI Grid -->
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-lbl">Total Inflow</div>
                <div class="stat-val" style="color: #10B981;">{format_currency(kpis.total_income, currency_symbol)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-lbl">Total Outflow</div>
                <div class="stat-val" style="color: #F87171;">{format_currency(kpis.total_expenses, currency_symbol)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-lbl">Net Saved</div>
                <div class="stat-val" style="color: {'#10B981' if kpis.net_savings >= 0 else '#EF4444'};">{format_currency(kpis.net_savings, currency_symbol)}</div>
            </div>
            <div class="stat-box">
                <div class="stat-lbl">Savings Rate</div>
                <div class="stat-val" style="color: #A78BFA;">{kpis.savings_rate:.1f}%</div>
            </div>
        </div>

        <!-- Overall Score Card -->
        <div class="card" style="display: flex; gap: 24px; align-items: center;">
            <div class="score-circle">
                {sc.total_score}
            </div>
            <div style="flex: 1;">
                <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF;">
                    Financial Health Grade: <span style="color: {sc.grade_color};">{sc.grade_label} ({sc.total_score}/100)</span>
                </div>
                <div style="font-size: 0.95rem; color: #CBD5E1; margin-top: 6px;">
                    {sc.summary_verdict}
                </div>
            </div>
        </div>

        <!-- Health Pillars -->
        <div class="card">
            <h3 style="margin-top: 0; color: #FFFFFF; font-size: 1.1rem;">Financial Health Pillars (Evaluation Breakdown)</h3>
            {pillars_html}
        </div>

        <!-- $100 Rule Breakdown -->
        <div class="card">
            <h3 style="margin-top: 0; color: #FFFFFF; font-size: 1.1rem;">The Dollar-for-Dollar Rule: Where Every {currency_symbol}100 Goes</h3>
            <p style="color: #CBD5E1; font-size: 0.92rem;">{hd.summary_text}</p>
            <div style="display: flex; border-radius: 8px; overflow: hidden; height: 18px; margin-top: 12px; background: #1E293B;">
                <div style="background: #6366F1; width: {max(2.0, hd.fixed_bills)}%;" title="Fixed Bills: {hd.fixed_bills}%"></div>
                <div style="background: #14B8A6; width: {max(2.0, hd.flexible_essentials)}%;" title="Essentials: {hd.flexible_essentials}%"></div>
                <div style="background: #EC4899; width: {max(2.0, hd.discretionary_fun)}%;" title="Fun & Dining: {hd.discretionary_fun}%"></div>
                <div style="background: #10B981; width: {max(2.0, max(0, hd.saved_pocket))}%;" title="Saved: {max(0, hd.saved_pocket)}%"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #94A3B8; margin-top: 8px;">
                <span>■ Fixed Bills: {hd.fixed_bills}%</span>
                <span>■ Essentials: {hd.flexible_essentials}%</span>
                <span>■ Fun & Dining: {hd.discretionary_fun}%</span>
                <span>■ Saved: {max(0, hd.saved_pocket)}%</span>
            </div>
        </div>

        <!-- Human Takeaways -->
        <div class="card">
            <h3 style="margin-top: 0; color: #FFFFFF; font-size: 1.1rem;">Key Takeaways & Actionable Tips</h3>
            {takeaways_html}
        </div>

        <!-- Forward Looking Outlook & Subscriptions -->
        <div class="card">
            <h3 style="margin-top: 0; color: #FFFFFF; font-size: 1.1rem;">What to Expect Next Month & Subscription Audit</h3>
            <p style="color: #CBD5E1; font-size: 0.9rem;"><b>Next Month Projection:</b> {report.outlook_plain_summary}</p>
            <p style="color: #CBD5E1; font-size: 0.9rem;"><b>Recurring Obligations:</b> {report.subscription_plain_summary}</p>
            <p style="color: #CBD5E1; font-size: 0.9rem;"><b>Surprise Outliers:</b> {report.outlier_plain_summary}</p>
        </div>

        <div style="text-align: center; font-size: 0.75rem; color: #64748B; margin-top: 24px;">
            SpendScope Personal Financial Intelligence Platform • Generated for private personal financial review.
        </div>
    </div>
</body>
</html>"""
    return html
