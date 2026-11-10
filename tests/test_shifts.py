"""
Tests for src/shifts.py (Spending Shift Analysis).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.categorization import TransactionCategorizer
from src.ingestion import load_and_validate_csv
from src.preprocessing import preprocess_transactions
from src.shifts import analyze_spending_shifts


def get_prepared_sample_data() -> pd.DataFrame:
    sample_path = Path("data/sample_transactions.csv")
    raw_df, val = load_and_validate_csv(sample_path)
    clean_df, _ = preprocess_transactions(raw_df, val.detected_columns)
    categorizer = TransactionCategorizer()
    return categorizer.categorize_dataframe(clean_df)


def test_analyze_spending_shifts():
    df = get_prepared_sample_data()
    # Sample data spans Nov 2023 - Apr 2024 (6 months)
    shift = analyze_spending_shifts(df, curr_month="2024-03", prev_month="2024-02")

    assert shift is not None
    assert shift.prev_month == "2024-02"
    assert shift.curr_month == "2024-03"
    assert shift.prev_total_expense > 0
    assert shift.curr_total_expense > 0
    assert len(shift.category_shifts) > 0
    assert len(shift.top_merchant_drivers) > 0

    # Verify narrative explanation is generated with real numbers
    assert "spending totaled" in shift.narrative_explanation
    assert shift.curr_month in shift.narrative_explanation

    # Verify waterfall structure
    assert not shift.waterfall_data.empty
    assert "step" in shift.waterfall_data.columns
    assert "amount" in shift.waterfall_data.columns


def test_analyze_spending_shifts_insufficient_data():
    single_month_df = pd.DataFrame({
        "year_month": ["2024-01", "2024-01"],
        "transaction_type": ["expense", "income"],
        "amount": [50.0, 100.0],
        "category": ["Dining & Food Delivery", "Income / Salary"],
        "merchant": ["Chipotle", "TechCorp"],
    })
    shift = analyze_spending_shifts(single_month_df)
    assert shift is None
