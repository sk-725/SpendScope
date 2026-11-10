"""
Tests for src/forecasting.py (Financial Outlook & Projections).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.categorization import TransactionCategorizer
from src.forecasting import generate_financial_outlook
from src.ingestion import load_and_validate_csv
from src.preprocessing import preprocess_transactions
from src.recurring import detect_recurring_expenses


def get_prepared_sample_data() -> pd.DataFrame:
    sample_path = Path("data/sample_transactions.csv")
    raw_df, val = load_and_validate_csv(sample_path)
    clean_df, _ = preprocess_transactions(raw_df, val.detected_columns)
    categorizer = TransactionCategorizer()
    return categorizer.categorize_dataframe(clean_df)


def test_generate_financial_outlook():
    df = get_prepared_sample_data()
    rec_items, _ = detect_recurring_expenses(df)
    forecast = generate_financial_outlook(df, recurring_items=rec_items)

    assert forecast is not None
    assert forecast.expected_total_spend > 0
    assert forecast.expected_fixed_commitments > 0
    assert forecast.conservative_spend_lower <= forecast.expected_total_spend
    assert forecast.aggressive_spend_upper >= forecast.expected_total_spend
    assert 0.0 <= forecast.confidence_score <= 1.0
    assert "Methodology" in forecast.methodology_text
    assert "Limitations" in forecast.limitations_text
    assert not forecast.timeline_df.empty
    assert "Projected Forecast" in forecast.timeline_df["type"].values


def test_forecast_insufficient_data():
    empty_df = pd.DataFrame()
    forecast = generate_financial_outlook(empty_df)
    assert forecast is None
