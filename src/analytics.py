"""
SpendScope - Expense Analytics Module.

Computes financial KPIs, income vs expense breakdowns, monthly velocity,
category distributions, daily spend timelines, and savings rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FinancialKPIs:
    """Core financial summary metrics for a selected timeframe."""
    total_income: float
    total_expenses: float
    net_savings: float
    savings_rate: float  # Percentage (0 - 100)
    transaction_count: int
    avg_daily_spend: float
    largest_expense: float
    largest_expense_desc: str


def calculate_financial_kpis(df: pd.DataFrame) -> FinancialKPIs:
    """
    Calculate high-level financial KPIs from categorized DataFrame.
    """
    income_mask = df["transaction_type"] == "income"
    expense_mask = df["transaction_type"] == "expense"

    total_income = float(df.loc[income_mask, "amount"].sum())
    total_expenses = float(df.loc[expense_mask, "amount"].sum())
    net_savings = total_income - total_expenses

    savings_rate = (net_savings / total_income * 100.0) if total_income > 0 else 0.0

    expense_df = df[expense_mask]
    transaction_count = len(df)

    if not expense_df.empty:
        max_idx = expense_df["amount"].idxmax()
        largest_expense = float(expense_df.loc[max_idx, "amount"])
        largest_expense_desc = str(expense_df.loc[max_idx, "description"])

        # Days range
        date_range = (df["date"].max() - df["date"].min()).days + 1
        avg_daily_spend = total_expenses / max(1, date_range)
    else:
        largest_expense = 0.0
        largest_expense_desc = "None"
        avg_daily_spend = 0.0

    return FinancialKPIs(
        total_income=total_income,
        total_expenses=total_expenses,
        net_savings=net_savings,
        savings_rate=savings_rate,
        transaction_count=transaction_count,
        avg_daily_spend=avg_daily_spend,
        largest_expense=largest_expense,
        largest_expense_desc=largest_expense_desc,
    )


def get_category_breakdown(df: pd.DataFrame, transaction_type: str = "expense") -> pd.DataFrame:
    """
    Compute spend distribution grouped by category with percentages and counts.
    """
    subset = df[df["transaction_type"] == transaction_type]
    if subset.empty:
        return pd.DataFrame(columns=["category", "total_amount", "percentage", "count", "color", "icon"])

    grouped = (
        subset.groupby("category")
        .agg(
            total_amount=("amount", "sum"),
            count=("transaction_id", "count"),
            color=("category_color", "first"),
            icon=("category_icon", "first"),
        )
        .reset_index()
    )

    total_sum = grouped["total_amount"].sum()
    grouped["percentage"] = (grouped["total_amount"] / total_sum * 100.0) if total_sum > 0 else 0.0
    return grouped.sort_values(by="total_amount", ascending=False).reset_index(drop=True)


def get_monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate monthly income, expense, and net savings trends.
    """
    if df.empty:
        return pd.DataFrame(columns=["year_month", "income", "expense", "net_savings"])

    monthly = (
        df.groupby(["year_month", "transaction_type"])["amount"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )

    if "income" not in monthly.columns:
        monthly["income"] = 0.0
    if "expense" not in monthly.columns:
        monthly["expense"] = 0.0

    monthly["net_savings"] = monthly["income"] - monthly["expense"]
    return monthly.sort_values(by="year_month").reset_index(drop=True)
