"""
SpendScope - Budget Management & Goal Tracking Module.

Compares actual category spending across monthly billing cycles against user-defined
budget caps, computing velocity, utilization percentages, and status alerts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class BudgetStatus:
    """Evaluation of spend vs monthly budget limit for a category."""
    category: str
    monthly_limit: float
    current_spend: float
    remaining: float
    percentage_used: float
    status: str  # "Under Budget", "Near Limit", "Exceeded"
    status_color: str  # Hex color for progress indicator


DEFAULT_SAMPLE_BUDGETS: Dict[str, float] = {
    "Housing & Rent": 1800.0,
    "Groceries & Supermarkets": 600.0,
    "Dining & Food Delivery": 300.0,
    "Utilities & Bills": 350.0,
    "Transportation & Fuel": 250.0,
    "Entertainment & Streaming": 120.0,
    "Shopping & Retail": 250.0,
}


def evaluate_budgets(
    df: pd.DataFrame,
    target_year_month: Optional[str] = None,
    budget_limits: Optional[Dict[str, float]] = None,
) -> List[BudgetStatus]:
    """
    Calculate budget utilization for each category within a selected month.

    Args:
        df: Categorized DataFrame containing transactions.
        target_year_month: YYYY-MM string (defaults to latest month in dataset).
        budget_limits: Dict of category -> monthly dollar cap.

    Returns:
        List of BudgetStatus records sorted by percentage used descending.
    """
    expense_df = df[df["transaction_type"] == "expense"].copy()
    if expense_df.empty:
        return []

    # Default to latest month if not specified
    if not target_year_month or target_year_month not in expense_df["year_month"].values:
        target_year_month = str(expense_df["year_month"].max())

    month_expenses = expense_df[expense_df["year_month"] == target_year_month]
    actual_spend_by_cat = month_expenses.groupby("category")["amount"].sum().to_dict()

    effective_limits = budget_limits or DEFAULT_SAMPLE_BUDGETS
    evaluations: List[BudgetStatus] = []

    for cat, limit in effective_limits.items():
        if limit <= 0:
            continue
        spent = float(actual_spend_by_cat.get(cat, 0.0))
        remaining = limit - spent
        pct = (spent / limit) * 100.0

        if pct > 100.0:
            status = "Exceeded"
            color = "#EF4444"  # Red
        elif pct >= 80.0:
            status = "Near Limit"
            color = "#F59E0B"  # Amber
        else:
            status = "Under Budget"
            color = "#10B981"  # Emerald Green

        evaluations.append(
            BudgetStatus(
                category=cat,
                monthly_limit=limit,
                current_spend=spent,
                remaining=remaining,
                percentage_used=round(pct, 1),
                status=status,
                status_color=color,
            )
        )

    # Sort highest utilization first
    evaluations.sort(key=lambda b: b.percentage_used, reverse=True)
    return evaluations
