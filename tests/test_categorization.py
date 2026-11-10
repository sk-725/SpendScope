"""
Tests for src/categorization.py
"""

import pandas as pd
from src.categorization import TransactionCategorizer


def test_merchant_extraction():
    categorizer = TransactionCategorizer()
    assert categorizer.extract_merchant("Starbucks Coffee #1042") == "Starbucks"
    assert categorizer.extract_merchant("Trader Joe's Groceries") == "Trader Joe's"
    assert categorizer.extract_merchant("Whole Foods Market") == "Whole Foods Market"
    assert categorizer.extract_merchant("Netflix.com Monthly Subscription") == "Netflix"
    assert categorizer.extract_merchant("Uber *Trip 4921") == "Uber"
    assert categorizer.extract_merchant("ConEdison Electric & Gas Utility") == "ConEdison"
    assert categorizer.extract_merchant("Apple Store Downtown - MacBook Pro M3") == "Apple Store"


def test_categorize_description():
    categorizer = TransactionCategorizer()
    # Groceries
    cat, conf = categorizer.categorize_description("Trader Joe's Groceries", "expense")
    assert cat == "Groceries & Supermarkets"
    assert conf >= 0.8

    # Dining
    cat, conf = categorizer.categorize_description("Chipotle Mexican Grill", "expense")
    assert cat == "Dining & Food Delivery"

    # Streaming
    cat, conf = categorizer.categorize_description("Spotify USA Monthly Sub", "expense")
    assert cat == "Entertainment & Streaming"

    # Utilities
    cat, conf = categorizer.categorize_description("ConEdison Electric & Gas", "expense")
    assert cat == "Utilities & Bills"

    # Housing
    cat, conf = categorizer.categorize_description("Highland Apartments Monthly Rent", "expense")
    assert cat == "Housing & Rent"

    # Income override
    cat, conf = categorizer.categorize_description("Salary Direct Deposit", "income")
    assert cat == "Income / Salary"
    assert conf >= 0.9

    # Unrecognized fallback
    cat, conf = categorizer.categorize_description("Random Unregistered Merchant XYZ", "expense")
    assert cat == "Other / Miscellaneous"
    assert conf < 0.5


def test_categorize_dataframe():
    categorizer = TransactionCategorizer()
    df = pd.DataFrame({
        "description": ["Trader Joe's Groceries", "Netflix.com", "Unknown Store"],
        "transaction_type": ["expense", "expense", "expense"],
        "raw_category": [None, None, None],
    })

    result_df = categorizer.categorize_dataframe(df)
    assert "category" in result_df.columns
    assert "category_confidence" in result_df.columns
    assert "merchant" in result_df.columns
    assert "category_icon" in result_df.columns
    assert "category_color" in result_df.columns

    assert result_df.iloc[0]["category"] == "Groceries & Supermarkets"
    assert result_df.iloc[0]["merchant"] == "Trader Joe's"
    assert result_df.iloc[1]["category"] == "Entertainment & Streaming"
    assert result_df.iloc[1]["merchant"] == "Netflix"


def test_custom_runtime_rule():
    categorizer = TransactionCategorizer()
    # Initially "CustomGizmo" would be miscellaneous
    cat_before, _ = categorizer.categorize_description("CustomGizmo Monthly", "expense")
    assert cat_before == "Other / Miscellaneous"

    # Add custom rule
    categorizer.add_runtime_rule("Shopping & Retail", "customgizmo")
    cat_after, conf = categorizer.categorize_description("CustomGizmo Monthly", "expense")
    assert cat_after == "Shopping & Retail"
    assert conf >= 0.85

