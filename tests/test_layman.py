"""
Tests for src/layman_intelligence.py (Human Translation & Financial Health Score).
"""

from pathlib import Path
import pandas as pd
import pytest

from src.analytics import calculate_financial_kpis
from src.anomaly_detection import ExpenseAnomalyDetector
from src.behavior import build_spending_behavior_profile
from src.categorization import TransactionCategorizer
from src.forecasting import generate_financial_outlook
from src.ingestion import load_and_validate_csv
from src.layman_intelligence import (
    compute_financial_health_score,
    compute_hundred_dollar_breakdown,
    generate_executive_html_briefing,
    generate_layman_report,
)
from src.preprocessing import preprocess_transactions
from src.recurring import detect_recurring_expenses, identify_subscription_leakage


def get_prepared_sample_data() -> pd.DataFrame:
    sample_path = Path("data/sample_transactions.csv")
    raw_df, val = load_and_validate_csv(sample_path)
    clean_df, _ = preprocess_transactions(raw_df, val.detected_columns)
    categorizer = TransactionCategorizer()
    cat_df = categorizer.categorize_dataframe(clean_df)
    detector = ExpenseAnomalyDetector()
    scored_df, _ = detector.fit_predict(cat_df)
    return scored_df


def test_compute_financial_health_score():
    df = get_prepared_sample_data()
    # Test healthy profile
    card = compute_financial_health_score(
        df=df,
        savings_rate=22.0,
        fixed_burden_pct=40.0,
        anomalous_spend_pct=2.0,
        recurring_redundancy_count=0,
        currency_symbol="$",
    )
    assert card.total_score >= 80
    assert card.grade_label in ["Excellent", "Good"]
    assert len(card.pillars) == 4
    for p in card.pillars:
        assert p.score > 0
        assert p.layman_explanation

    # Test distressed profile
    bad_card = compute_financial_health_score(
        df=df,
        savings_rate=-5.0,
        fixed_burden_pct=85.0,
        anomalous_spend_pct=25.0,
        recurring_redundancy_count=2,
        currency_symbol="$",
    )
    assert bad_card.total_score < 50
    assert bad_card.grade_label == "Needs Attention"


def test_compute_hundred_dollar_breakdown():
    breakdown = compute_hundred_dollar_breakdown(
        total_income=10000.0,
        total_expense=8000.0,
        fixed_total=3500.0,
        flexible_total=2500.0,
        discretionary_total=2000.0,
        currency_symbol="$",
    )
    assert breakdown.fixed_bills == 35.0
    assert breakdown.flexible_essentials == 25.0
    assert breakdown.discretionary_fun == 20.0
    assert breakdown.saved_pocket == 20.0
    assert "$100" in breakdown.summary_text


def test_generate_layman_report_and_html_briefing():
    df = get_prepared_sample_data()
    kpis = calculate_financial_kpis(df)
    profile = build_spending_behavior_profile(df)
    rec_items, _ = detect_recurring_expenses(df)
    leakage = identify_subscription_leakage(rec_items, df)
    outlook = generate_financial_outlook(df, recurring_items=rec_items)

    report = generate_layman_report(
        df=df,
        kpis=kpis,
        behavior_profile=profile,
        recurring_items=rec_items,
        recurring_leakage=leakage,
        outlook=outlook,
        currency_symbol="$",
    )

    assert report.health_scorecard.total_score > 0
    assert len(report.top_takeaways) >= 2
    assert "essential living costs" in report.top_habit_summary.lower()
    assert report.subscription_plain_summary

    html = generate_executive_html_briefing(
        report=report,
        statement_name="sample_transactions.csv",
        date_range_str="Nov 2023 - Apr 2024",
        kpis=kpis,
        currency_symbol="$",
    )
    assert "<!DOCTYPE html>" in html
    assert "SpendScope" in html
    assert "Financial Health Grade" in html
    assert "@media print" in html
