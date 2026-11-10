"""
SpendScope - Recurring Expense & Subscription Leakage Intelligence Module.

Identifies recurring payments, subscriptions, and regular bills by analyzing
merchant consistency, amount variance, and temporal intervals between transactions.
Additionally identifies concurrent service redundancies and potential subscription leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class RecurringItem:
    """Metadata for an identified recurring expense or subscription."""
    merchant: str
    category: str
    cadence: str  # e.g., "Monthly", "Weekly", "Bi-Weekly", "Quarterly"
    expected_amount: float
    is_variable: bool
    occurrence_count: int
    last_date: str
    next_projected_date: str
    annual_cost: float
    confidence: float  # 0.0 to 1.0


@dataclass
class RecurringSummary:
    """Aggregated commitment overview."""
    total_recurring_items: int
    monthly_recurring_spend: float
    annual_recurring_spend: float
    top_recurring_merchants: List[str]


@dataclass
class SubscriptionRedundancy:
    """Identified cluster of concurrent services within the same functional domain."""
    domain: str  # e.g., "Digital Entertainment & Media", "Cloud & Hosting"
    merchants: List[str]
    count: int
    combined_monthly_cost: float
    combined_annual_cost: float
    insight_note: str


@dataclass
class SubscriptionLeakageAnalysis:
    """Intelligence on committed outflows and potential leakage."""
    summary: RecurringSummary
    items: List[RecurringItem]
    redundancies: List[SubscriptionRedundancy]
    total_redundant_annual_cost: float
    fixed_obligation_ratio_pct: float  # Recurring spend as % of total monthly expenses


def detect_cadence(day_intervals: List[int]) -> Tuple[Optional[str], float]:
    """
    Determine payment cadence and confidence from a list of day intervals.

    Returns:
        (cadence_label, confidence_score)
    """
    if not day_intervals:
        return None, 0.0

    median_days = float(np.median(day_intervals))

    # Weekly: 6 - 8 days
    if 6 <= median_days <= 8:
        return "Weekly", 0.90
    # Bi-Weekly: 13 - 16 days
    elif 13 <= median_days <= 16:
        return "Bi-Weekly", 0.88
    # Monthly: 27 - 33 days
    elif 27 <= median_days <= 33:
        return "Monthly", 0.95
    # Quarterly: 85 - 95 days
    elif 85 <= median_days <= 95:
        return "Quarterly", 0.85
    # Semi-annual / Annual
    elif 350 <= median_days <= 380:
        return "Annual", 0.80

    return "Irregular", 0.40


def detect_recurring_expenses(
    df: pd.DataFrame,
    min_occurrences: int = 2,
    amount_tolerance_pct: float = 0.15,
) -> Tuple[List[RecurringItem], RecurringSummary]:
    """
    Scan transactions for recurring patterns based on merchant, amount, and cadence.

    Args:
        df: Categorized DataFrame containing expenses.
        min_occurrences: Minimum count of transactions to qualify.
        amount_tolerance_pct: Max coefficient of variation in amount (e.g. 15%).

    Returns:
        Tuple of (List of RecurringItem, RecurringSummary).
    """
    expense_df = df[df["transaction_type"] == "expense"].copy()
    if expense_df.empty:
        return [], RecurringSummary(0, 0.0, 0.0, [])

    expense_df = expense_df.sort_values(by="date")
    recurring_items: List[RecurringItem] = []

    # Group by merchant
    for merchant, group in expense_df.groupby("merchant"):
        if len(group) < min_occurrences:
            continue

        amounts = group["amount"].values
        mean_amt = float(np.mean(amounts))
        std_amt = float(np.std(amounts))
        cv = std_amt / (mean_amt + 1e-9)

        # Calculate time intervals between successive dates
        dates = pd.to_datetime(group["date"]).sort_values().values
        intervals = [
            int((dates[i] - dates[i - 1]) / np.timedelta64(1, "D"))
            for i in range(1, len(dates))
        ]

        cadence, cadence_conf = detect_cadence(intervals)
        if cadence is None or cadence == "Irregular":
            continue

        # Check amount stability: fixed or slightly variable
        is_variable = cv > 0.02
        if cv > amount_tolerance_pct and cadence != "Monthly":
            # Highly variable amounts on non-monthly schedule are usually not subscriptions
            continue

        # Project next billing date
        last_dt = pd.to_datetime(group["date"].max())
        if cadence == "Weekly":
            next_dt = last_dt + pd.Timedelta(days=7)
            annual_cost = mean_amt * 52.0
        elif cadence == "Bi-Weekly":
            next_dt = last_dt + pd.Timedelta(days=14)
            annual_cost = mean_amt * 26.0
        elif cadence == "Monthly":
            next_dt = last_dt + pd.DateOffset(months=1)
            annual_cost = mean_amt * 12.0
        elif cadence == "Quarterly":
            next_dt = last_dt + pd.DateOffset(months=3)
            annual_cost = mean_amt * 4.0
        else:
            next_dt = last_dt + pd.DateOffset(years=1)
            annual_cost = mean_amt

        category = str(group["category"].iloc[0])
        overall_conf = round(cadence_conf * (1.0 - min(0.3, cv)), 2)

        recurring_items.append(
            RecurringItem(
                merchant=str(merchant),
                category=category,
                cadence=cadence,
                expected_amount=round(mean_amt, 2),
                is_variable=is_variable,
                occurrence_count=len(group),
                last_date=last_dt.strftime("%Y-%m-%d"),
                next_projected_date=next_dt.strftime("%Y-%m-%d"),
                annual_cost=round(annual_cost, 2),
                confidence=overall_conf,
            )
        )

    # Sort items by annual cost descending
    recurring_items.sort(key=lambda x: x.annual_cost, reverse=True)

    total_monthly = sum(
        item.annual_cost / 12.0 for item in recurring_items
    )
    total_annual = sum(item.annual_cost for item in recurring_items)
    top_merchants = [item.merchant for item in recurring_items[:5]]

    summary = RecurringSummary(
        total_recurring_items=len(recurring_items),
        monthly_recurring_spend=round(total_monthly, 2),
        annual_recurring_spend=round(total_annual, 2),
        top_recurring_merchants=top_merchants,
    )

    return recurring_items, summary


def identify_subscription_leakage(
    items: List[RecurringItem],
    df: pd.DataFrame,
    currency_symbol: str = "$",
) -> SubscriptionLeakageAnalysis:
    """
    Examine detected recurring items for potential overlap, category clustering, and leakage.
    """
    sym = currency_symbol
    redundancies: List[SubscriptionRedundancy] = []
    redundant_annual_sum = 0.0

    # Cluster by category to identify concurrent services
    entertainment_items = [it for it in items if it.category == "Entertainment & Streaming"]
    if len(entertainment_items) >= 2:
        merchants = [it.merchant for it in entertainment_items]
        combined_mo = sum(it.annual_cost / 12.0 for it in entertainment_items)
        combined_yr = sum(it.annual_cost for it in entertainment_items)
        redundancies.append(
            SubscriptionRedundancy(
                domain="Digital Media & Streaming",
                merchants=merchants,
                count=len(merchants),
                combined_monthly_cost=round(combined_mo, 2),
                combined_annual_cost=round(combined_yr, 2),
                insight_note=(
                    f"You have {len(merchants)} concurrent active digital media services ({', '.join(merchants)}) "
                    f"costing {sym}{combined_mo:,.2f}/mo ({sym}{combined_yr:,.2f}/yr). Consider rotating services."
                ),
            )
        )
        redundant_annual_sum += combined_yr

    # Compute recurring spend as % of total monthly expenses
    expense_mask = df["transaction_type"] == "expense"
    total_exp = float(df.loc[expense_mask, "amount"].sum())
    
    # Months span
    if "year_month" in df.columns:
        num_months = max(1, df["year_month"].nunique())
    else:
        num_months = 1
        
    avg_mo_exp = (total_exp / num_months) if num_months > 0 else 1.0
    mo_rec_spend = sum(it.annual_cost / 12.0 for it in items)
    fixed_ratio = (mo_rec_spend / avg_mo_exp * 100.0) if avg_mo_exp > 0 else 0.0

    summary = RecurringSummary(
        total_recurring_items=len(items),
        monthly_recurring_spend=round(mo_rec_spend, 2),
        annual_recurring_spend=round(sum(it.annual_cost for it in items), 2),
        top_recurring_merchants=[it.merchant for it in items[:5]],
    )

    return SubscriptionLeakageAnalysis(
        summary=summary,
        items=items,
        redundancies=redundancies,
        total_redundant_annual_cost=round(redundant_annual_sum, 2),
        fixed_obligation_ratio_pct=round(min(100.0, fixed_ratio), 1),
    )
