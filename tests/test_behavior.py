"""
Tests for src/behavior.py (Spending Behaviour Intelligence).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.behavior import (
    analyze_day_of_week,
    analyze_merchant_concentration,
    analyze_structural_composition,
    analyze_weekday_weekend,
    build_spending_behavior_profile,
)
from src.categorization import TransactionCategorizer
from src.ingestion import load_and_validate_csv
from src.preprocessing import preprocess_transactions


def get_prepared_sample_data() -> pd.DataFrame:
    sample_path = Path("data/sample_transactions.csv")
    raw_df, val = load_and_validate_csv(sample_path)
    clean_df, _ = preprocess_transactions(raw_df, val.detected_columns)
    categorizer = TransactionCategorizer()
    return categorizer.categorize_dataframe(clean_df)


def test_analyze_weekday_weekend():
    df = get_prepared_sample_data()
    expense_df = df[df["transaction_type"] == "expense"]
    ww = analyze_weekday_weekend(expense_df)

    assert ww.weekday_total > 0
    assert ww.weekend_total > 0
    assert round(ww.weekday_pct + ww.weekend_pct) == 100
    assert ww.weekend_intensity_ratio > 0
    assert ww.top_weekday_category != "None"


def test_analyze_day_of_week():
    df = get_prepared_sample_data()
    expense_df = df[df["transaction_type"] == "expense"]
    dow_df, peak, lowest = analyze_day_of_week(expense_df)

    assert len(dow_df) == 7
    assert "amount" in dow_df.columns
    assert "avg_txn" in dow_df.columns
    assert peak in dow_df["day_name"].values
    assert lowest in dow_df["day_name"].values


def test_analyze_merchant_concentration():
    df = get_prepared_sample_data()
    expense_df = df[df["transaction_type"] == "expense"]
    mc = analyze_merchant_concentration(expense_df)

    assert mc.total_unique_merchants > 10
    assert 0 < mc.top_3_share_pct <= 100
    assert 0 < mc.hhi_index <= 10000
    assert mc.concentration_level in ["Diversified", "Moderate Concentration", "High Concentration"]
    assert mc.pareto_80_merchant_count > 0
    assert not mc.top_merchants_table.empty


def test_build_spending_behavior_profile():
    df = get_prepared_sample_data()
    profile = build_spending_behavior_profile(df)

    assert profile.weekday_weekend.weekday_total > 0
    assert len(profile.behavioral_traits) >= 2
    for trait in profile.behavioral_traits:
        assert trait.title
        assert trait.evidence
        assert trait.archetype
