"""
Tests for src/ingestion.py
"""

import io
from pathlib import Path

import pytest

from src.ingestion import (
    CANONICAL_AMOUNT,
    CANONICAL_CREDIT,
    CANONICAL_DATE,
    CANONICAL_DEBIT,
    CANONICAL_DESCRIPTION,
    detect_columns,
    detect_delimiter,
    load_and_validate_csv,
    read_csv_safely,
    validate_raw_dataframe,
)


def test_detect_columns_standard():
    headers = ["Date", "Description", "Amount", "Category"]
    mapping = detect_columns(headers)
    assert mapping[CANONICAL_DATE] == "Date"
    assert mapping[CANONICAL_DESCRIPTION] == "Description"
    assert mapping[CANONICAL_AMOUNT] == "Amount"
    assert mapping["category"] == "Category"


def test_detect_columns_synonyms():
    headers = ["Posting Date", "Payee", "Total", "Tag"]
    mapping = detect_columns(headers)
    assert mapping[CANONICAL_DATE] == "Posting Date"
    assert mapping[CANONICAL_DESCRIPTION] == "Payee"
    assert mapping[CANONICAL_AMOUNT] == "Total"


def test_detect_columns_split_debit_credit():
    headers = ["Txn_Date", "Narrative", "Withdrawal", "Deposit"]
    mapping = detect_columns(headers)
    assert mapping[CANONICAL_DATE] == "Txn_Date"
    assert mapping[CANONICAL_DESCRIPTION] == "Narrative"
    assert mapping[CANONICAL_DEBIT] == "Withdrawal"
    assert mapping[CANONICAL_CREDIT] == "Deposit"


def test_detect_delimiter():
    csv_comma = "Date,Description,Amount\n2024-01-01,Test,10"
    assert detect_delimiter(csv_comma) == ","

    csv_semi = "Date;Description;Amount\n2024-01-01;Test;10"
    assert detect_delimiter(csv_semi) == ";"

    csv_tab = "Date\tDescription\tAmount\n2024-01-01\tTest\t10"
    assert detect_delimiter(csv_tab) == "\t"


def test_validate_empty_file():
    empty_csv = io.StringIO("")
    df, result = load_and_validate_csv(empty_csv)
    assert df is None
    assert not result.is_valid
    assert any("empty" in err.lower() for err in result.errors)


def test_validate_missing_columns():
    csv_data = io.StringIO("ColA,ColB\n1,2\n3,4")
    df, result = load_and_validate_csv(csv_data)
    assert df is not None
    assert not result.is_valid
    assert any("date" in err.lower() for err in result.errors)
    assert any("description" in err.lower() for err in result.errors)
    assert any("value" in err.lower() or "amount" in err.lower() for err in result.errors)


def test_load_sample_transactions_file():
    sample_path = Path("data/sample_transactions.csv")
    assert sample_path.exists(), "Sample transactions CSV must exist."
    df, result = load_and_validate_csv(sample_path)
    assert df is not None
    assert result.is_valid
    assert result.total_raw_rows > 100
    assert CANONICAL_DATE in result.detected_columns
    assert CANONICAL_DESCRIPTION in result.detected_columns
    assert CANONICAL_AMOUNT in result.detected_columns
