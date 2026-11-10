"""
Tests for anomaly detection, recurring expenses, and SQLite database persistence.
"""

from pathlib import Path
import tempfile
import pandas as pd

from src.anomaly_detection import ExpenseAnomalyDetector
from src.categorization import TransactionCategorizer
from src.database import (
    add_custom_rule,
    delete_custom_rule,
    delete_session,
    get_custom_rules,
    list_sessions,
    load_session,
    save_session,
)
from src.ingestion import load_and_validate_csv
from src.preprocessing import preprocess_transactions
from src.recurring import detect_recurring_expenses, identify_subscription_leakage


def get_prepared_data():
    sample_path = Path("data/sample_transactions.csv")
    raw_df, val = load_and_validate_csv(sample_path)
    clean_df, _ = preprocess_transactions(raw_df, val.detected_columns)
    categorizer = TransactionCategorizer()
    return categorizer.categorize_dataframe(clean_df)


def test_anomaly_detection():
    df = get_prepared_data()
    detector = ExpenseAnomalyDetector(contamination=0.05, random_state=42)
    scored_df, summary = detector.fit_predict(df)

    assert "is_anomaly" in scored_df.columns
    assert "anomaly_score" in scored_df.columns
    assert "anomaly_severity" in scored_df.columns
    assert "anomaly_tag" in scored_df.columns
    assert "anomaly_explanation" in scored_df.columns

    assert summary.anomalies_detected > 0
    assert summary.high_severity_count > 0

    # Specifically verify the MacBook Pro anomaly was caught with contextual tag
    macbook_row = scored_df[scored_df["description"].str.contains("MacBook Pro", case=False)]
    assert not macbook_row.empty
    assert bool(macbook_row["is_anomaly"].iloc[0]) is True
    assert macbook_row["anomaly_severity"].iloc[0] == "High"
    assert "First-Time" in macbook_row["anomaly_tag"].iloc[0] or "Scale" in macbook_row["anomaly_tag"].iloc[0]


def test_recurring_detection():
    df = get_prepared_data()
    items, summary = detect_recurring_expenses(df, min_occurrences=2)

    assert summary.total_recurring_items > 0
    assert summary.monthly_recurring_spend > 0
    assert len(items) > 0

    merchants = [it.merchant for it in items]
    # Highland Apartments, Netflix, Planet Fitness, ConEdison, Spotify should be detected
    assert any("Netflix" in m for m in merchants)
    assert any("Highland Apartments" in m for m in merchants)
    assert any("Planet Fitness" in m for m in merchants)


def test_subscription_leakage_analysis():
    df = get_prepared_data()
    items, _ = detect_recurring_expenses(df, min_occurrences=2)
    leakage = identify_subscription_leakage(items, df)

    assert leakage.summary.total_recurring_items > 0
    assert leakage.fixed_obligation_ratio_pct > 0
    assert len(leakage.redundancies) >= 1
    assert leakage.redundancies[0].domain == "Digital Media & Streaming"
    assert "Digital media services" in leakage.redundancies[0].insight_note or "concurrent" in leakage.redundancies[0].insight_note



def test_sqlite_database_lifecycle():
    df = get_prepared_data()
    detector = ExpenseAnomalyDetector(contamination=0.05, random_state=42)
    scored_df, _ = detector.fit_predict(df)

    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_spendscope.db"
        session_id = save_session(
            df=scored_df,
            session_name="Unit Test Session",
            file_name="sample_transactions.csv",
            db_path=test_db,
        )

        assert session_id.startswith("SES-")

        sessions = list_sessions(db_path=test_db)
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == session_id
        assert sessions[0]["total_rows"] == len(scored_df)

        loaded_df = load_session(session_id, db_path=test_db)
        assert loaded_df is not None
        assert len(loaded_df) == len(scored_df)
        assert "is_anomaly" in loaded_df.columns

        deleted = delete_session(session_id, db_path=test_db)
        assert deleted is True
        assert len(list_sessions(db_path=test_db)) == 0


def test_custom_rules_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_rules.db"

        rule_id = add_custom_rule("Entertainment & Streaming", "crunchyroll", db_path=test_db)
        assert rule_id > 0

        rules = get_custom_rules(db_path=test_db)
        assert len(rules) == 1
        assert rules[0]["pattern"] == "crunchyroll"

        categorizer = TransactionCategorizer.from_database(db_path=test_db)
        cat, conf = categorizer.categorize_description("Crunchyroll Premium Sub", "expense")
        assert cat == "Entertainment & Streaming"
        assert conf >= 0.85

        del_ok = delete_custom_rule(rule_id, db_path=test_db)
        assert del_ok is True
        assert len(get_custom_rules(db_path=test_db)) == 0

