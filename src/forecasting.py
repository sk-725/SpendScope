"""
SpendScope - Financial Outlook & Historical Forecasting Module.

Generates basic baseline expenditure projections using weighted historical moving
averages, structural commitment floors, and empirical variance bounds.
Includes explicit methodology disclosures and analytical limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.recurring import RecurringItem


@dataclass
class MonthlyForecast:
    """Projected financial trajectory for the upcoming monthly billing cycle."""
    forecast_month: str
    expected_fixed_commitments: float
    expected_discretionary_spend: float
    expected_total_spend: float
    conservative_spend_lower: float  # -1 std dev bound
    aggressive_spend_upper: float    # +1 std dev bound
    expected_income: float
    expected_net_savings: float
    expected_savings_rate: float
    confidence_score: float  # 0.0 - 1.0
    history_months_count: int
    burn_rate_avg: float
    methodology_text: str
    limitations_text: str
    timeline_df: pd.DataFrame  # month, type ('Historical', 'Projected'), spend, lower, upper


METHODOLOGY_NOTE = (
    "Methodology: Multi-component baseline model combining structural fixed obligations "
    "(identified recurring commitments) with an exponentially weighted 3-month discretionary "
    "run-rate. Prediction intervals represent empirical 1-standard-deviation historical variance bounds."
)

LIMITATIONS_NOTE = (
    "Limitations & Caveats: Short-term personal finance projections cannot anticipate unannounced life events, "
    "annual insurance premiums, tax liabilities, or discretionary travel. With less than 12-24 months of data, "
    "seasonal cycles cannot be fully controlled. This projection assumes consistent income and routine behavior."
)


def generate_financial_outlook(
    df: pd.DataFrame,
    recurring_items: Optional[List[RecurringItem]] = None,
    currency_symbol: str = "$",
) -> Optional[MonthlyForecast]:
    """
    Produce an evidence-based next-month spending projection and historical bounds.

    Args:
        df: Clean canonical transactions DataFrame.
        recurring_items: List of detected recurring obligations.
        currency_symbol: Active display currency symbol.

    Returns:
        MonthlyForecast dataclass or None if dataset has insufficient historical records.
    """
    if df.empty or "year_month" not in df.columns:
        return None

    monthly_summary = (
        df.groupby(["year_month", "transaction_type"])["amount"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )

    if "expense" not in monthly_summary.columns or len(monthly_summary) < 1:
        return None

    if "income" not in monthly_summary.columns:
        monthly_summary["income"] = 0.0

    monthly_expenses = monthly_summary["expense"].values
    monthly_incomes = monthly_summary["income"].values
    history_len = len(monthly_summary)

    # Calculate average burn rate and standard deviation
    avg_burn = float(np.mean(monthly_expenses))
    std_burn = float(np.std(monthly_expenses)) if history_len > 1 else (avg_burn * 0.15)
    avg_income = float(np.mean(monthly_incomes)) if len(monthly_incomes) > 0 else 0.0

    # Fixed obligations floor
    if recurring_items:
        fixed_floor = sum(it.annual_cost / 12.0 for it in recurring_items)
    else:
        # Heuristic floor from essential categories
        fixed_mask = df["category"].isin(["Housing & Rent", "Utilities & Bills"])
        fixed_floor = float(df[fixed_mask]["amount"].sum() / max(1, history_len))

    # Weight recent months more heavily if >= 3 months exist
    if history_len >= 3:
        weights = np.linspace(0.5, 1.0, history_len)
        weights /= weights.sum()
        weighted_expense = float(np.sum(monthly_expenses * weights))
        confidence = min(0.85, 0.50 + (history_len * 0.05))
    elif history_len == 2:
        weighted_expense = float(monthly_expenses[-1] * 0.6 + monthly_expenses[-2] * 0.4)
        confidence = 0.55
    else:
        weighted_expense = float(monthly_expenses[0])
        confidence = 0.35

    # Ensure forecast respects fixed commitment floor
    discretionary_projected = max(0.0, weighted_expense - fixed_floor)
    expected_total = fixed_floor + discretionary_projected

    lower_bound = max(fixed_floor, expected_total - std_burn)
    upper_bound = expected_total + std_burn

    # Project next month label (YYYY-MM)
    last_ym = monthly_summary["year_month"].iloc[-1]
    try:
        last_dt = pd.to_datetime(f"{last_ym}-01")
        next_dt = last_dt + pd.DateOffset(months=1)
        next_ym = next_dt.strftime("%Y-%m")
    except Exception:
        next_ym = "Next Month"

    # Savings expectations
    exp_income = avg_income
    exp_savings = exp_income - expected_total
    exp_savings_rate = (exp_savings / exp_income * 100.0) if exp_income > 0 else 0.0

    # Construct timeline dataframe for Plotly
    timeline_records = []
    for _, row in monthly_summary.iterrows():
        timeline_records.append({
            "month": row["year_month"],
            "type": "Historical Actual",
            "spend": float(row["expense"]),
            "lower_bound": float(row["expense"]),
            "upper_bound": float(row["expense"]),
        })

    timeline_records.append({
        "month": next_ym,
        "type": "Projected Forecast",
        "spend": round(expected_total, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
    })

    timeline_df = pd.DataFrame(timeline_records)

    return MonthlyForecast(
        forecast_month=next_ym,
        expected_fixed_commitments=round(fixed_floor, 2),
        expected_discretionary_spend=round(discretionary_projected, 2),
        expected_total_spend=round(expected_total, 2),
        conservative_spend_lower=round(lower_bound, 2),
        aggressive_spend_upper=round(upper_bound, 2),
        expected_income=round(exp_income, 2),
        expected_net_savings=round(exp_savings, 2),
        expected_savings_rate=round(exp_savings_rate, 1),
        confidence_score=round(confidence, 2),
        history_months_count=history_len,
        burn_rate_avg=round(avg_burn, 2),
        methodology_text=METHODOLOGY_NOTE,
        limitations_text=LIMITATIONS_NOTE,
        timeline_df=timeline_df,
    )
