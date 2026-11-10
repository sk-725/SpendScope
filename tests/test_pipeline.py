"""
End-to-end integration test for SpendScope core ingestion and preprocessing pipeline.
"""

from pathlib import Path
import pytest

from src.categorization import TransactionCategorizer
from src.ingestion import load_and_validate_csv
from src.preprocessing import preprocess_transactions


def test_end_to_end_sample_pipeline():
    sample_file = Path("data/sample_transactions.csv")
    assert sample_file.exists(), "Sample transactions file must be present."

    # 1. Ingestion & Validation
    raw_df, val_result = load_and_validate_csv(sample_file)
    assert val_result.is_valid, f"Validation failed: {val_result.errors}"
    assert raw_df is not None
    assert len(raw_df) > 100

    # 2. Preprocessing
    clean_df, prep_report = preprocess_transactions(raw_df, val_result.detected_columns)
    assert prep_report.final_rows == len(raw_df)
    assert prep_report.dropped_rows == 0
    assert prep_report.income_count > 0
    assert prep_report.expense_count > 0
    assert prep_report.total_income > 0
    assert prep_report.total_expense > 0

    # 3. Categorization
    categorizer = TransactionCategorizer()
    categorized_df = categorizer.categorize_dataframe(clean_df)

    # Validate output schema
    required_cols = [
        "transaction_id",
        "date",
        "description",
        "merchant",
        "category",
        "category_icon",
        "category_color",
        "amount",
        "transaction_type",
        "net_amount",
        "year",
        "month",
        "year_month",
        "day_name",
        "is_weekend",
    ]
    for col in required_cols:
        assert col in categorized_df.columns, f"Missing canonical column '{col}'"

    # Validate category distributions
    category_counts = categorized_df["category"].value_counts()
    assert "Groceries & Supermarkets" in category_counts
    assert "Dining & Food Delivery" in category_counts
    assert "Utilities & Bills" in category_counts
    assert "Housing & Rent" in category_counts
    assert "Income / Salary" in category_counts

    # High accuracy check on sample dataset: < 5% unclassified
    unclassified_count = (categorized_df["category"] == "Other / Miscellaneous").sum()
    unclassified_pct = unclassified_count / len(categorized_df)
    assert unclassified_pct < 0.05, f"Too many unclassified transactions: {unclassified_pct:.1%}"
