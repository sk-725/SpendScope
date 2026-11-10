"""
SpendScope - Spending Behaviour Intelligence Module.

Analyzes behavioral dimensions of personal expenditures:
- Weekday vs Weekend spending dynamics and leisure intensity
- Day-of-week distribution patterns
- Merchant concentration (Herfindahl-Hirschman Index & Pareto distribution)
- Structural expenditure composition (Fixed vs Discretionary vs Flexible)
- Evidence-based behavioral trait profiles derived from empirical metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Category classification taxonomies
FIXED_ESSENTIAL_CATEGORIES = {
    "Housing & Rent",
    "Utilities & Bills",
    "Financial & Fees",
}

DISCRETIONARY_CATEGORIES = {
    "Dining & Food Delivery",
    "Entertainment & Streaming",
    "Shopping & Retail",
    "Travel & Lodging",
}

FLEXIBLE_ESSENTIAL_CATEGORIES = {
    "Groceries & Supermarkets",
    "Transportation & Fuel",
    "Health & Fitness",
    "Other / Miscellaneous",
}

DAYS_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


@dataclass
class WeekdayWeekendMetrics:
    """Quantitative comparison between weekday and weekend spending."""
    weekday_total: float
    weekend_total: float
    weekday_pct: float
    weekend_pct: float
    weekday_txn_count: int
    weekend_txn_count: int
    avg_weekday_txn: float
    avg_weekend_txn: float
    avg_daily_weekday: float
    avg_daily_weekend: float
    weekend_intensity_ratio: float  # weekend daily / weekday daily
    top_weekday_category: str
    top_weekend_category: str


@dataclass
class MerchantConcentrationMetrics:
    """Measures of vendor concentration and dependency."""
    total_unique_merchants: int
    top_3_share_pct: float
    top_5_share_pct: float
    top_10_share_pct: float
    hhi_index: float  # Herfindahl-Hirschman Index (0 - 10000)
    concentration_level: str  # "Diversified", "Moderate", "High Concentration"
    pareto_80_merchant_count: int  # Number of merchants generating 80% of spend
    top_merchants_table: pd.DataFrame  # rank, merchant, category, amount, pct, count


@dataclass
class StructuralComposition:
    """Breakdown of spending into structural buckets."""
    fixed_total: float
    fixed_pct: float
    discretionary_total: float
    discretionary_pct: float
    flexible_total: float
    flexible_pct: float


@dataclass
class BehavioralTrait:
    """An empirical, non-psychological spending pattern backed by numerical evidence."""
    title: str
    archetype: str  # e.g., "Outflow Timing", "Vendor Structure", "Flexibility"
    description: str
    evidence: str
    significance: str  # "High", "Moderate", "Neutral"


@dataclass
class SpendingBehaviorProfile:
    """Comprehensive analytical profile of spending habits."""
    weekday_weekend: WeekdayWeekendMetrics
    day_of_week_distribution: pd.DataFrame  # day_name, amount, pct, count, avg_txn
    merchant_concentration: MerchantConcentrationMetrics
    composition: StructuralComposition
    peak_spending_day: str
    lowest_spending_day: str
    behavioral_traits: List[BehavioralTrait]


def analyze_weekday_weekend(expense_df: pd.DataFrame) -> WeekdayWeekendMetrics:
    """Analyze differences between weekday and weekend expenditures."""
    if expense_df.empty:
        return WeekdayWeekendMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, "None", "None")

    weekend_mask = expense_df["is_weekend"] == True
    weekday_mask = ~weekend_mask

    weekday_exp = expense_df[weekday_mask]
    weekend_exp = expense_df[weekend_mask]

    weekday_total = float(weekday_exp["amount"].sum())
    weekend_total = float(weekend_exp["amount"].sum())
    total = weekday_total + weekend_total

    weekday_pct = (weekday_total / total * 100.0) if total > 0 else 0.0
    weekend_pct = (weekend_total / total * 100.0) if total > 0 else 0.0

    weekday_cnt = len(weekday_exp)
    weekend_cnt = len(weekend_exp)

    avg_weekday_txn = (weekday_total / weekday_cnt) if weekday_cnt > 0 else 0.0
    avg_weekend_txn = (weekend_total / weekend_cnt) if weekend_cnt > 0 else 0.0

    # Number of calendar weekdays vs weekend days in dataset range
    if "date" in expense_df.columns and not expense_df.empty:
        date_min = expense_df["date"].min()
        date_max = expense_df["date"].max()
        all_dates = pd.date_range(date_min, date_max)
        num_weekdays = int((all_dates.dayofweek < 5).sum())
        num_weekends = int((all_dates.dayofweek >= 5).sum())
    else:
        num_weekdays = 5
        num_weekends = 2

    daily_weekday = (weekday_total / max(1, num_weekdays))
    daily_weekend = (weekend_total / max(1, num_weekends))
    intensity = (daily_weekend / daily_weekday) if daily_weekday > 0 else 1.0

    top_weekday_cat = (
        weekday_exp.groupby("category")["amount"].sum().idxmax()
        if not weekday_exp.empty else "None"
    )
    top_weekend_cat = (
        weekend_exp.groupby("category")["amount"].sum().idxmax()
        if not weekend_exp.empty else "None"
    )

    return WeekdayWeekendMetrics(
        weekday_total=round(weekday_total, 2),
        weekend_total=round(weekend_total, 2),
        weekday_pct=round(weekday_pct, 1),
        weekend_pct=round(weekend_pct, 1),
        weekday_txn_count=weekday_cnt,
        weekend_txn_count=weekend_cnt,
        avg_weekday_txn=round(avg_weekday_txn, 2),
        avg_weekend_txn=round(avg_weekend_txn, 2),
        avg_daily_weekday=round(daily_weekday, 2),
        avg_daily_weekend=round(daily_weekend, 2),
        weekend_intensity_ratio=round(intensity, 2),
        top_weekday_category=str(top_weekday_cat),
        top_weekend_category=str(top_weekend_cat),
    )


def analyze_day_of_week(expense_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    """Calculate day-of-week spending totals, frequencies, and identify peak/trough days."""
    if expense_df.empty:
        empty_df = pd.DataFrame(columns=["day_name", "amount", "pct", "count", "avg_txn"])
        return empty_df, "None", "None"

    grouped = (
        expense_df.groupby("day_name")
        .agg(
            amount=("amount", "sum"),
            count=("transaction_id", "count"),
        )
        .reindex(DAYS_ORDER)
        .fillna(0.0)
        .reset_index()
    )

    total_amt = grouped["amount"].sum()
    grouped["pct"] = (grouped["amount"] / total_amt * 100.0) if total_amt > 0 else 0.0
    grouped["avg_txn"] = np.where(
        grouped["count"] > 0,
        grouped["amount"] / grouped["count"],
        0.0,
    )
    grouped["pct"] = grouped["pct"].round(1)
    grouped["avg_txn"] = grouped["avg_txn"].round(2)

    peak_day = str(grouped.loc[grouped["amount"].idxmax(), "day_name"])
    lowest_day = str(grouped.loc[grouped["amount"].idxmin(), "day_name"])

    return grouped, peak_day, lowest_day


def analyze_merchant_concentration(expense_df: pd.DataFrame) -> MerchantConcentrationMetrics:
    """Analyze vendor concentration using HHI and Pareto distribution."""
    if expense_df.empty:
        empty_table = pd.DataFrame(columns=["rank", "merchant", "category", "amount", "pct", "count"])
        return MerchantConcentrationMetrics(0, 0, 0, 0, 0, "Diversified", 0, empty_table)

    total_expense = float(expense_df["amount"].sum())
    grouped = (
        expense_df.groupby("merchant")
        .agg(
            category=("category", "first"),
            amount=("amount", "sum"),
            count=("transaction_id", "count"),
        )
        .sort_values(by="amount", ascending=False)
        .reset_index()
    )

    grouped["pct"] = (grouped["amount"] / total_expense * 100.0) if total_expense > 0 else 0.0
    grouped["cumulative_pct"] = grouped["pct"].cumsum()
    grouped["rank"] = range(1, len(grouped) + 1)

    # Compute Herfindahl-Hirschman Index (HHI)
    # HHI = sum((market_share_percentage)^2)
    shares = grouped["pct"].values
    hhi = float(np.sum(shares ** 2))

    if hhi > 2500:
        level = "High Concentration"
    elif hhi >= 1500:
        level = "Moderate Concentration"
    else:
        level = "Diversified"

    top_3_pct = float(grouped.head(3)["pct"].sum())
    top_5_pct = float(grouped.head(5)["pct"].sum())
    top_10_pct = float(grouped.head(10)["pct"].sum())

    # Pareto 80% threshold: how many vendors make up 80% of spend?
    pareto_idx = (grouped["cumulative_pct"] >= 80.0).idxmax() if not grouped.empty else 0
    pareto_count = int(grouped.loc[pareto_idx, "rank"]) if not grouped.empty else 0

    return MerchantConcentrationMetrics(
        total_unique_merchants=len(grouped),
        top_3_share_pct=round(top_3_pct, 1),
        top_5_share_pct=round(top_5_pct, 1),
        top_10_share_pct=round(top_10_pct, 1),
        hhi_index=round(hhi, 1),
        concentration_level=level,
        pareto_80_merchant_count=pareto_count,
        top_merchants_table=grouped[["rank", "merchant", "category", "amount", "pct", "count"]].head(15),
    )


def analyze_structural_composition(expense_df: pd.DataFrame) -> StructuralComposition:
    """Decompose expenditures into Fixed, Discretionary, and Flexible essentials."""
    if expense_df.empty:
        return StructuralComposition(0, 0, 0, 0, 0, 0)

    total = float(expense_df["amount"].sum())

    fixed_spend = float(
        expense_df[expense_df["category"].isin(FIXED_ESSENTIAL_CATEGORIES)]["amount"].sum()
    )
    disc_spend = float(
        expense_df[expense_df["category"].isin(DISCRETIONARY_CATEGORIES)]["amount"].sum()
    )
    flex_spend = float(
        expense_df[expense_df["category"].isin(FLEXIBLE_ESSENTIAL_CATEGORIES)]["amount"].sum()
    )

    f_pct = (fixed_spend / total * 100.0) if total > 0 else 0.0
    d_pct = (disc_spend / total * 100.0) if total > 0 else 0.0
    fl_pct = (flex_spend / total * 100.0) if total > 0 else 0.0

    return StructuralComposition(
        fixed_total=round(fixed_spend, 2),
        fixed_pct=round(f_pct, 1),
        discretionary_total=round(disc_spend, 2),
        discretionary_pct=round(d_pct, 1),
        flexible_total=round(flex_spend, 2),
        flexible_pct=round(fl_pct, 1),
    )


def derive_behavioral_traits(
    ww: WeekdayWeekendMetrics,
    mc: MerchantConcentrationMetrics,
    comp: StructuralComposition,
    peak_day: str,
) -> List[BehavioralTrait]:
    """
    Generate evidence-based behavioral traits from empirical metrics.
    Does not use speculative psychology; derives strictly from observed numbers.
    """
    traits: List[BehavioralTrait] = []

    # 1. Timing Archetype
    if ww.weekend_intensity_ratio >= 1.5:
        traits.append(
            BehavioralTrait(
                title="Weekend-Skewed Outflow",
                archetype="Timing Dynamics",
                description="Daily spending during weekends significantly exceeds typical weekday velocity.",
                evidence=f"Weekend daily spend is {ww.weekend_intensity_ratio:.1f}x weekday burn (${ww.avg_daily_weekend:,.2f}/day vs ${ww.avg_daily_weekday:,.2f}/day), led by {ww.top_weekend_category}.",
                significance="High",
            )
        )
    elif ww.weekend_intensity_ratio <= 0.7:
        traits.append(
            BehavioralTrait(
                title="Weekday-Concentrated Routine",
                archetype="Timing Dynamics",
                description="Expenditures are primarily clustered during working weekdays with quiet weekends.",
                evidence=f"Weekdays capture {ww.weekday_pct:.1f}% of outflows, with peak outflow occurring on {peak_day}.",
                significance="Moderate",
            )
        )
    else:
        traits.append(
            BehavioralTrait(
                title="Even Temporal Velocity",
                archetype="Timing Dynamics",
                description="Spending is distributed relatively evenly throughout the week without extreme weekend spikes.",
                evidence=f"Weekend-to-weekday intensity ratio is balanced at {ww.weekend_intensity_ratio:.2f}x (baseline: 1.0x).",
                significance="Neutral",
            )
        )

    # 2. Vendor Structure Archetype
    if mc.hhi_index > 2500:
        traits.append(
            BehavioralTrait(
                title="High Merchant Concentration",
                archetype="Vendor Footprint",
                description="A small handful of key merchants absorb the majority of your outgoing capital.",
                evidence=f"Top 3 vendors command {mc.top_3_share_pct:.1f}% of total outflows; HHI is {mc.hhi_index:,.0f} (above the 2,500 concentrated index threshold).",
                significance="High",
            )
        )
    elif mc.hhi_index < 1200:
        traits.append(
            BehavioralTrait(
                title="Diversified Vendor Dispersion",
                archetype="Vendor Footprint",
                description="Expenditures are broadly dispersed across numerous independent merchants and service providers.",
                evidence=f"HHI index is {mc.hhi_index:,.0f} with {mc.total_unique_merchants} unique merchants across the billing period.",
                significance="Moderate",
            )
        )

    # 3. Structural Composition Archetype
    if comp.fixed_pct >= 55.0:
        traits.append(
            BehavioralTrait(
                title="Fixed-Heavy Structural Commitments",
                archetype="Structural Flexibility",
                description="Inflexible structural commitments (Housing, Utilities, Loans) represent the core burden of monthly cash flow.",
                evidence=f"Fixed obligations constitute {comp.fixed_pct:.1f}% of expenses (${comp.fixed_total:,.2f}), leaving {comp.discretionary_pct:.1f}% for discretionary choices.",
                significance="High",
            )
        )
    elif comp.discretionary_pct >= 35.0:
        traits.append(
            BehavioralTrait(
                title="High Discretionary Elasticity",
                archetype="Structural Flexibility",
                description="A substantial portion of spending is discretionary and amenable to immediate budgetary adjustment.",
                evidence=f"Discretionary categories (Dining, Shopping, Entertainment, Travel) absorb {comp.discretionary_pct:.1f}% (${comp.discretionary_total:,.2f}) of outflows.",
                significance="High",
            )
        )
    else:
        traits.append(
            BehavioralTrait(
                title="Balanced Structural Baseline",
                archetype="Structural Flexibility",
                description="Spending maintains a healthy balance between fixed obligations, flexible essentials, and discretionary choices.",
                evidence=f"Fixed: {comp.fixed_pct:.1f}%, Flexible: {comp.flexible_pct:.1f}%, Discretionary: {comp.discretionary_pct:.1f}%.",
                significance="Neutral",
            )
        )

    return traits


def build_spending_behavior_profile(df: pd.DataFrame) -> SpendingBehaviorProfile:
    """
    Main entry point for generating comprehensive Spending Behaviour Intelligence.
    """
    expense_df = df[df["transaction_type"] == "expense"].copy()

    ww = analyze_weekday_weekend(expense_df)
    dow_df, peak_day, lowest_day = analyze_day_of_week(expense_df)
    mc = analyze_merchant_concentration(expense_df)
    comp = analyze_structural_composition(expense_df)
    traits = derive_behavioral_traits(ww, mc, comp, peak_day)

    return SpendingBehaviorProfile(
        weekday_weekend=ww,
        day_of_week_distribution=dow_df,
        merchant_concentration=mc,
        composition=comp,
        peak_spending_day=peak_day,
        lowest_spending_day=lowest_day,
        behavioral_traits=traits,
    )
