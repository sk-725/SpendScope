"""
Tests for src/preprocessing.py
"""

import pandas as pd
import pytest

from src.ingestion import CANONICAL_AMOUNT, CANONICAL_DATE, CANONICAL_DESCRIPTION, CANONICAL_TYPE
from src.preprocessing import (
    clean_currency_string,
    clean_description,
    detect_currency_symbol,
    format_currency,
    parse_dates_robustly,
    preprocess_transactions,
)


def test_detect_currency_symbol():
    euro_series = pd.Series(["€45.00", "€120.50", "15.00"])
    assert detect_currency_symbol(euro_series) == "€"

    gbp_series = pd.Series(["£80.00", "£12.50"])
    assert detect_currency_symbol(gbp_series) == "£"

    inr_series = pd.Series(["₹1,500.00", "₹450.00"])
    assert detect_currency_symbol(inr_series) == "₹"

    yen_series = pd.Series(["¥3000", "¥1500"])
    assert detect_currency_symbol(yen_series) == "¥"

    plain_series = pd.Series(["45.00", "120.50"])
    assert detect_currency_symbol(plain_series) == "$"


def test_format_currency():
    assert format_currency(1234.56, "$") == "$1,234.56"
    assert format_currency(1234.56, "€") == "€1,234.56"
    assert format_currency(1234.56, "₹") == "₹1,234.56"
    assert format_currency(-45.5, "£") == "-£45.50"
    assert format_currency(0.0, "$") == "$0.00"



def test_clean_currency_string():
    assert clean_currency_string("$1,234.56") == 1234.56
    assert clean_currency_string("€50.00") == 50.00
    assert clean_currency_string("£99.99") == 99.99
    assert clean_currency_string("(150.25)") == -150.25
    assert clean_currency_string("-84.20") == -84.20
    assert clean_currency_string("75.10-") == -75.10
    assert clean_currency_string("+200.00") == 200.00
    assert clean_currency_string("0.0") == 0.0
    assert clean_currency_string(None) is None
    assert clean_currency_string("N/A") is None


def test_clean_description():
    assert clean_description("   Trader   Joe's   ") == "Trader Joe's"
    assert clean_description("Whole Foods\t\tMarket") == "Whole Foods Market"
    assert clean_description(None) == "Unknown Description"
    assert clean_description("") == "Unknown Description"


def test_parse_dates_robustly():
    dates = pd.Series(["2024-01-15", "2024/02/20", "15-03-2024", "invalid_date"])
    parsed = parse_dates_robustly(dates)
    assert pd.notna(parsed[0])
    assert parsed[0].year == 2024
    assert parsed[0].month == 1
    assert parsed[0].day == 15
    assert pd.isna(parsed[3])


def test_preprocess_transactions_pipeline():
    raw_data = {
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03", "invalid-date"],
        "Description": ["Salary Deposit", "Grocery Store", "Coffee", "Broken Row"],
        "Amount": ["4500.00", "84.50", "4.75", "10.00"],
        "Type": ["income", "expense", "expense", "expense"],
    }
    raw_df = pd.DataFrame(raw_data)
    mapping = {
        CANONICAL_DATE: "Date",
        CANONICAL_DESCRIPTION: "Description",
        CANONICAL_AMOUNT: "Amount",
        CANONICAL_TYPE: "Type",
    }

    clean_df, report = preprocess_transactions(raw_df, mapping)

    # 1 row with invalid date should be dropped
    assert report.initial_rows == 4
    assert report.final_rows == 3
    assert report.dropped_rows == 1
    assert report.income_count == 1
    assert report.expense_count == 2
    assert report.total_income == 4500.00
    assert report.total_expense == 84.50 + 4.75

    # Check temporal features
    assert "year" in clean_df.columns
    assert "month" in clean_df.columns
    assert "year_month" in clean_df.columns
    assert "day_name" in clean_df.columns
    assert "is_weekend" in clean_df.columns
    assert "quarter" in clean_df.columns
    assert "transaction_id" in clean_df.columns
    assert clean_df["transaction_id"].str.startswith("TXN-").all()


def test_preprocess_split_debit_credit():
    raw_data = {
        "Date": ["2024-02-01", "2024-02-02"],
        "Description": ["Paycheck", "Gas Station"],
        "Debit": ["", "$45.00"],
        "Credit": ["$3000.00", ""],
    }
    raw_df = pd.DataFrame(raw_data)
    mapping = {
        "date": "Date",
        "description": "Description",
        "debit": "Debit",
        "credit": "Credit",
    }

    clean_df, report = preprocess_transactions(raw_df, mapping)
    assert report.final_rows == 2
    assert report.income_count == 1
    assert report.expense_count == 1
    assert clean_df.iloc[0]["transaction_type"] == "income"
    assert clean_df.iloc[1]["transaction_type"] == "expense"


def test_preprocess_non_dollar_currency():
    raw_data = {
        "Date": ["2024-03-01", "2024-03-02"],
        "Description": ["Paris Bakery", "Berlin Train"],
        "Amount": ["€24.50", "€89.00"],
        "Type": ["expense", "expense"],
    }
    raw_df = pd.DataFrame(raw_data)
    mapping = {
        "date": "Date",
        "description": "Description",
        "amount": "Amount",
        "type": "Type",
    }

    clean_df, report = preprocess_transactions(raw_df, mapping)
    assert report.currency_symbol == "€"
    assert clean_df["currency_symbol"].iloc[0] == "€"
    assert clean_df["amount"].iloc[0] == 24.50

