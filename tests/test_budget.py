"""
Tests for src/budget.py and SQLite budget limits.
"""

from pathlib import Path
import tempfile
import pandas as pd

from src.budget import evaluate_budgets
from src.database import delete_category_budget, get_category_budgets, set_category_budget


def test_evaluate_budgets():
    # Synthetic month transactions
    df = pd.DataFrame({
        "year_month": ["2024-04", "2024-04", "2024-04", "2024-03"],
        "category": [
            "Dining & Food Delivery",
            "Dining & Food Delivery",
            "Groceries & Supermarkets",
            "Dining & Food Delivery",
        ],
        "amount": [120.0, 150.0, 300.0, 50.0],
        "transaction_type": ["expense", "expense", "expense", "expense"],
    })

    limits = {
        "Dining & Food Delivery": 200.0,     # spent 270 -> exceeded
        "Groceries & Supermarkets": 500.0,   # spent 300 -> 60% under budget
    }

    evals = evaluate_budgets(df, target_year_month="2024-04", budget_limits=limits)
    assert len(evals) == 2

    dining_eval = next(e for e in evals if e.category == "Dining & Food Delivery")
    assert dining_eval.current_spend == 270.0
    assert dining_eval.percentage_used == 135.0
    assert dining_eval.status == "Exceeded"
    assert dining_eval.remaining == -70.0

    groc_eval = next(e for e in evals if e.category == "Groceries & Supermarkets")
    assert groc_eval.current_spend == 300.0
    assert groc_eval.percentage_used == 60.0
    assert groc_eval.status == "Under Budget"


def test_sqlite_budget_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_budget.db"

        # Set limits
        set_category_budget("Dining & Food Delivery", 350.0, db_path=test_db)
        set_category_budget("Groceries & Supermarkets", 600.0, db_path=test_db)

        budgets = get_category_budgets(db_path=test_db)
        assert budgets["Dining & Food Delivery"] == 350.0
        assert budgets["Groceries & Supermarkets"] == 600.0

        # Update limit
        set_category_budget("Dining & Food Delivery", 400.0, db_path=test_db)
        budgets_updated = get_category_budgets(db_path=test_db)
        assert budgets_updated["Dining & Food Delivery"] == 400.0

        # Delete limit
        deleted = delete_category_budget("Dining & Food Delivery", db_path=test_db)
        assert deleted is True
        budgets_after = get_category_budgets(db_path=test_db)
        assert "Dining & Food Delivery" not in budgets_after
