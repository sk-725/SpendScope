"""
SpendScope - Smart Spending Insights Module.

Generates rule-based personal finance insights, month-over-month shifts,
budgetary warnings, and actionable recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FinancialInsight:
    """Represents a generated analytical insight."""
    type: str  # "success", "warning", "info", "danger"
    title: str
    message: str
    impact: str  # e.g., "High", "Medium", "Low"
    category: Optional[str] = None


def generate_spending_insights(
    df: pd.DataFrame,
    monthly_trends_df: Optional[pd.DataFrame] = None,
    currency_symbol: Optional[str] = None,
) -> List[FinancialInsight]:
    """
    Produce intelligent rule-based financial insights from analyzed transactions.
    """
    insights: List[FinancialInsight] = []
    if df.empty:
        return insights

    sym = currency_symbol or (df["currency_symbol"].iloc[0] if "currency_symbol" in df.columns and not df.empty else "$")

    expense_mask = df["transaction_type"] == "expense"
    income_mask = df["transaction_type"] == "income"

    total_income = float(df.loc[income_mask, "amount"].sum())
    total_expense = float(df.loc[expense_mask, "amount"].sum())
    net_savings = total_income - total_expense

    # 1. Savings Rate Analysis
    if total_income > 0:
        savings_rate = (net_savings / total_income) * 100.0
        if savings_rate >= 30.0:
            insights.append(
                FinancialInsight(
                    type="success",
                    title="Healthy Savings Rate",
                    message=f"You are retaining {savings_rate:.1f}% of your total earnings (saved {sym}{net_savings:,.2f}). This exceeds the 20% standard benchmark.",
                    impact="High",
                )
            )
        elif savings_rate > 0:
            insights.append(
                FinancialInsight(
                    type="info",
                    title="Moderate Savings Rate",
                    message=f"Your current savings rate is {savings_rate:.1f}% ({sym}{net_savings:,.2f} retained). Consider reducing discretionary categories to reach 20%.",
                    impact="Medium",
                )
            )
        else:
            deficit = abs(net_savings)
            insights.append(
                FinancialInsight(
                    type="danger",
                    title="Deficit Alert: Expenses Exceed Income",
                    message=f"Total expenditures exceed income by {sym}{deficit:,.2f}. Review high-ticket purchases and recurring obligations to balance cash flow.",
                    impact="High",
                )
            )

    # 2. Category Concentration Check
    expense_df = df[expense_mask]
    if not expense_df.empty:
        cat_sums = expense_df.groupby("category")["amount"].sum()
        top_cat = cat_sums.idxmax()
        top_cat_amt = float(cat_sums.max())
        top_cat_pct = (top_cat_amt / total_expense) * 100.0 if total_expense > 0 else 0.0

        if top_cat_pct >= 40.0:
            insights.append(
                FinancialInsight(
                    type="warning",
                    title=f"High Spend Concentration in {top_cat}",
                    message=f"{top_cat} consumes {top_cat_pct:.1f}% ({sym}{top_cat_amt:,.2f}) of your total expenses. A single category absorbing over 40% limits financial flexibility.",
                    impact="Medium",
                    category=top_cat,
                )
            )

    # 3. Discretionary vs Essential Dining/Coffee Check
    dining_spend = float(expense_df[expense_df["category"] == "Dining & Food Delivery"]["amount"].sum())
    if dining_spend > 0 and total_expense > 0:
        dining_pct = (dining_spend / total_expense) * 100.0
        if dining_pct > 15.0:
            insights.append(
                FinancialInsight(
                    type="warning",
                    title="Elevated Dining & Delivery Spend",
                    message=f"Dining and takeout account for {sym}{dining_spend:,.2f} ({dining_pct:.1f}% of total outflow). Shifting 25% to home meal preparation could save approximately {sym}{dining_spend * 0.25:,.2f}.",
                    impact="Medium",
                    category="Dining & Food Delivery",
                )
            )

    # 4. Month-over-Month Velocity Analysis
    if monthly_trends_df is not None and len(monthly_trends_df) >= 2:
        recent = monthly_trends_df.iloc[-1]
        previous = monthly_trends_df.iloc[-2]
        rec_exp = float(recent.get("expense", 0.0))
        prev_exp = float(previous.get("expense", 0.0))

        if prev_exp > 0:
            pct_change = ((rec_exp - prev_exp) / prev_exp) * 100.0
            if pct_change > 20.0:
                insights.append(
                    FinancialInsight(
                        type="warning",
                        title=f"Spending Surge in {recent['year_month']}",
                        message=f"Expenses rose by {pct_change:.1f}% compared to {previous['year_month']} (from {sym}{prev_exp:,.2f} to {sym}{rec_exp:,.2f}).",
                        impact="High",
                    )
                )
            elif pct_change < -15.0:
                insights.append(
                    FinancialInsight(
                        type="success",
                        title=f"Spending Reduction in {recent['year_month']}",
                        message=f"Expenses decreased by {abs(pct_change):.1f}% compared to {previous['year_month']} (from {sym}{prev_exp:,.2f} to {sym}{rec_exp:,.2f}).",
                        impact="Medium",
                    )
                )

    # 5. Weekend Spending Pattern
    if not expense_df.empty and "is_weekend" in expense_df.columns:
        weekend_spend = float(expense_df[expense_df["is_weekend"]]["amount"].sum())
        weekend_pct = (weekend_spend / total_expense) * 100.0 if total_expense > 0 else 0.0
        if weekend_pct >= 40.0:
            insights.append(
                FinancialInsight(
                    type="info",
                    title="Weekend Outflow Clustering",
                    message=f"{weekend_pct:.1f}% of your expenditures happen on Saturdays and Sundays. Weekend activities represent a major driver of discretionary variance.",
                    impact="Low",
                )
            )

    return insights
