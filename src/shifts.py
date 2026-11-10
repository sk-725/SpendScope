"""
SpendScope - Spending Shift Analysis Module.

Performs month-over-month variance decomposition, identifies primary category
and merchant change drivers, and computes mathematical explanations of financial shifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class CategoryShift:
    """Month-over-month variance metrics for an individual spending category."""
    category: str
    prev_amount: float
    curr_amount: float
    abs_change: float
    pct_change: float
    direction: str  # "Increase", "Decrease", "Unchanged"
    contribution_share_pct: float  # Share of total gross change


@dataclass
class MerchantShift:
    """Month-over-month variance metrics for an individual merchant."""
    merchant: str
    category: str
    prev_amount: float
    curr_amount: float
    abs_change: float
    status: str  # "Surged", "New Vendor", "Reduced", "No Activity"


@dataclass
class MonthOverMonthShiftAnalysis:
    """Comprehensive month-over-month expenditure shift intelligence."""
    prev_month: str
    curr_month: str
    prev_total_expense: float
    curr_total_expense: float
    net_expense_change: float
    pct_expense_change: float
    prev_total_income: float
    curr_total_income: float
    net_income_change: float
    prev_savings_rate: float
    curr_savings_rate: float
    savings_rate_change_pts: float  # Percentage points change
    category_shifts: List[CategoryShift]
    top_expansion_categories: List[CategoryShift]
    top_contraction_categories: List[CategoryShift]
    top_merchant_drivers: List[MerchantShift]
    narrative_explanation: str
    waterfall_data: pd.DataFrame  # step, label, amount, type


def analyze_spending_shifts(
    df: pd.DataFrame,
    curr_month: Optional[str] = None,
    prev_month: Optional[str] = None,
    currency_symbol: str = "$",
) -> Optional[MonthOverMonthShiftAnalysis]:
    """
    Decompose month-over-month spending shifts into category and merchant drivers.

    Args:
        df: Categorized DataFrame containing transactions.
        curr_month: Target YYYY-MM string (defaults to most recent month).
        prev_month: Baseline YYYY-MM string (defaults to month immediately preceding curr_month).
        currency_symbol: Symbol for formatted explanations.

    Returns:
        MonthOverMonthShiftAnalysis dataclass, or None if fewer than 2 months available.
    """
    if df.empty or "year_month" not in df.columns:
        return None

    available_months = sorted(df["year_month"].dropna().unique())
    if len(available_months) < 2:
        return None

    if curr_month is None or curr_month not in available_months:
        curr_month = available_months[-1]

    # Resolve baseline month
    curr_idx = available_months.index(curr_month)
    if prev_month is None or prev_month not in available_months:
        if curr_idx > 0:
            prev_month = available_months[curr_idx - 1]
        else:
            prev_month = available_months[1]
            curr_month, prev_month = prev_month, curr_month

    sym = currency_symbol

    # Filter data for both months
    prev_df = df[df["year_month"] == prev_month]
    curr_df = df[df["year_month"] == curr_month]

    # Totals
    prev_exp = float(prev_df[prev_df["transaction_type"] == "expense"]["amount"].sum())
    curr_exp = float(curr_df[curr_df["transaction_type"] == "expense"]["amount"].sum())
    net_exp_change = curr_exp - prev_exp
    pct_exp_change = ((curr_exp - prev_exp) / prev_exp * 100.0) if prev_exp > 0 else 0.0

    prev_inc = float(prev_df[prev_df["transaction_type"] == "income"]["amount"].sum())
    curr_inc = float(curr_df[curr_df["transaction_type"] == "income"]["amount"].sum())
    net_inc_change = curr_inc - prev_inc

    prev_sav_rate = ((prev_inc - prev_exp) / prev_inc * 100.0) if prev_inc > 0 else 0.0
    curr_sav_rate = ((curr_inc - curr_exp) / curr_inc * 100.0) if curr_inc > 0 else 0.0
    sav_rate_change_pts = curr_sav_rate - prev_sav_rate

    # Category variance decomposition
    prev_cat_sums = prev_df[prev_df["transaction_type"] == "expense"].groupby("category")["amount"].sum().to_dict()
    curr_cat_sums = curr_df[curr_df["transaction_type"] == "expense"].groupby("category")["amount"].sum().to_dict()

    all_cats = sorted(set(prev_cat_sums.keys()).union(set(curr_cat_sums.keys())))
    raw_cat_shifts: List[CategoryShift] = []
    gross_abs_diff_sum = 0.0

    for cat in all_cats:
        p_val = float(prev_cat_sums.get(cat, 0.0))
        c_val = float(curr_cat_sums.get(cat, 0.0))
        diff = c_val - p_val
        p_pct = ((diff / p_val) * 100.0) if p_val > 0 else (100.0 if c_val > 0 else 0.0)
        direction = "Increase" if diff > 0.01 else ("Decrease" if diff < -0.01 else "Unchanged")
        gross_abs_diff_sum += abs(diff)

        raw_cat_shifts.append(
            CategoryShift(
                category=cat,
                prev_amount=round(p_val, 2),
                curr_amount=round(c_val, 2),
                abs_change=round(diff, 2),
                pct_change=round(p_pct, 1),
                direction=direction,
                contribution_share_pct=0.0,  # Computed below
            )
        )

    # Compute contribution shares
    for item in raw_cat_shifts:
        item.contribution_share_pct = round(
            (abs(item.abs_change) / gross_abs_diff_sum * 100.0) if gross_abs_diff_sum > 0 else 0.0,
            1,
        )

    # Sort categories by absolute change descending
    raw_cat_shifts.sort(key=lambda x: abs(x.abs_change), reverse=True)

    expansions = [c for c in raw_cat_shifts if c.abs_change > 0]
    contractions = [c for c in raw_cat_shifts if c.abs_change < 0]

    # Merchant variance decomposition
    prev_merch_sums = prev_df[prev_df["transaction_type"] == "expense"].groupby("merchant")["amount"].sum().to_dict()
    curr_merch_sums = curr_df[curr_df["transaction_type"] == "expense"].groupby("merchant")["amount"].sum().to_dict()
    all_merchants = sorted(set(prev_merch_sums.keys()).union(set(curr_merch_sums.keys())))

    merchant_shifts: List[MerchantShift] = []
    for merch in all_merchants:
        p_amt = float(prev_merch_sums.get(merch, 0.0))
        c_amt = float(curr_merch_sums.get(merch, 0.0))
        diff = c_amt - p_amt

        if abs(diff) < 5.0:  # Ignore trivial fluctuations under $5
            continue

        if p_amt == 0.0 and c_amt > 0.0:
            status = "New Vendor"
        elif c_amt == 0.0 and p_amt > 0.0:
            status = "No Activity"
        elif diff > 0:
            status = "Surged"
        else:
            status = "Reduced"

        # Determine category for this merchant
        cat_match = curr_df[curr_df["merchant"] == merch]["category"]
        if cat_match.empty:
            cat_match = prev_df[prev_df["merchant"] == merch]["category"]
        cat_name = str(cat_match.iloc[0]) if not cat_match.empty else "General"

        merchant_shifts.append(
            MerchantShift(
                merchant=merch,
                category=cat_name,
                prev_amount=round(p_amt, 2),
                curr_amount=round(c_amt, 2),
                abs_change=round(diff, 2),
                status=status,
            )
        )

    merchant_shifts.sort(key=lambda m: abs(m.abs_change), reverse=True)
    top_merchants = merchant_shifts[:10]

    # Build Waterfall Data
    waterfall_records = [
        {"step": 1, "label": f"{prev_month} Total", "amount": prev_exp, "type": "total"}
    ]
    step_num = 2
    for c in raw_cat_shifts:
        if abs(c.abs_change) >= 10.0:
            waterfall_records.append({
                "step": step_num,
                "label": c.category,
                "amount": c.abs_change,
                "type": "relative",
            })
            step_num += 1

    waterfall_records.append({
        "step": step_num,
        "label": f"{curr_month} Total",
        "amount": curr_exp,
        "type": "total",
    })
    waterfall_df = pd.DataFrame(waterfall_records)

    # Generate Evidence-Based Narrative Explanation
    sign = "+" if net_exp_change >= 0 else "-"
    verb = "increased" if net_exp_change >= 0 else "decreased"
    narrative_parts = [
        f"**{curr_month}** spending totaled **{sym}{curr_exp:,.2f}**, which {verb} by **{sign}{sym}{abs(net_exp_change):,.2f}** (**{sign}{abs(pct_exp_change):.1f}%**) compared to **{prev_month}** ({sym}{prev_exp:,.2f})."
    ]

    if expansions:
        top_exp = expansions[0]
        narrative_parts.append(
            f"The primary upward driver was **{top_exp.category}**, which rose by **+{sym}{top_exp.abs_change:,.2f}** (+{top_exp.pct_change:.1f}%), contributing {top_exp.contribution_share_pct:.1f}% of total gross movement."
        )
        if len(expansions) > 1:
            second_exp = expansions[1]
            narrative_parts.append(
                f"Secondary expansion occurred in **{second_exp.category}** (+{sym}{second_exp.abs_change:,.2f})."
            )

    if contractions:
        top_con = contractions[0]
        narrative_parts.append(
            f"In contrast, the largest budgetary relief came from **{top_con.category}**, which dropped by **-{sym}{abs(top_con.abs_change):,.2f}** ({top_con.pct_change:.1f}%)."
        )

    # Savings rate context
    sav_verb = "improved" if sav_rate_change_pts >= 0 else "softened"
    narrative_parts.append(
        f"The net savings rate {sav_verb} by **{abs(sav_rate_change_pts):.1f} percentage points** (from {prev_sav_rate:.1f}% to {curr_sav_rate:.1f}%)."
    )

    full_narrative = " ".join(narrative_parts)

    return MonthOverMonthShiftAnalysis(
        prev_month=prev_month,
        curr_month=curr_month,
        prev_total_expense=round(prev_exp, 2),
        curr_total_expense=round(curr_exp, 2),
        net_expense_change=round(net_exp_change, 2),
        pct_expense_change=round(pct_exp_change, 1),
        prev_total_income=round(prev_inc, 2),
        curr_total_income=round(curr_inc, 2),
        net_income_change=round(net_inc_change, 2),
        prev_savings_rate=round(prev_sav_rate, 1),
        curr_savings_rate=round(curr_sav_rate, 1),
        savings_rate_change_pts=round(sav_rate_change_pts, 1),
        category_shifts=raw_cat_shifts,
        top_expansion_categories=expansions[:5],
        top_contraction_categories=contractions[:5],
        top_merchant_drivers=top_merchants,
        narrative_explanation=full_narrative,
        waterfall_data=waterfall_df,
    )
