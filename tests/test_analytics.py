"""
Tests for src/analytics.py and src/insights.py
"""

from pathlib import Path
import pandas as pd

from src.analytics import calculate_financial_kpis, get_category_breakdown, get_monthly_trends
from src.categorization import TransactionCategorizer
from src.ingestion import load_and_validate_csv
from src.insights import generate_spending_insights
from src.preprocessing import preprocess_transactions


def get_sample_categorized_df() -> pd.DataFrame:
    sample_path = Path("data/sample_transactions.csv")
    raw_df, val = load_and_validate_csv(sample_path)
    clean_df, _ = preprocess_transactions(raw_df, val.detected_columns)
    categorizer = TransactionCategorizer()
    return categorizer.categorize_dataframe(clean_df)


def test_calculate_financial_kpis():
    df = get_sample_categorized_df()
    kpis = calculate_financial_kpis(df)
    assert kpis.total_income > 0
    assert kpis.total_expenses > 0
    assert kpis.net_savings > 0
    assert 0 <= kpis.savings_rate <= 100
    assert kpis.largest_expense > 1000.0


def test_get_category_breakdown():
    df = get_sample_categorized_df()
    breakdown = get_category_breakdown(df)
    assert not breakdown.empty
    assert "total_amount" in breakdown.columns
    assert "percentage" in breakdown.columns
    assert round(breakdown["percentage"].sum()) == 100


def test_get_monthly_trends():
    df = get_sample_categorized_df()
    trends = get_monthly_trends(df)
    assert len(trends) >= 5
    assert "income" in trends.columns
    assert "expense" in trends.columns
    assert "net_savings" in trends.columns


def test_generate_spending_insights():
    df = get_sample_categorized_df()
    trends = get_monthly_trends(df)
    insights = generate_spending_insights(df, trends)
    assert len(insights) > 0
    types = [i.type for i in insights]
    assert any(t in ["success", "warning", "info"] for t in types)
