"""
SpendScope - Database Module (SQLite).

Provides persistent storage for uploaded transaction batches, analysis sessions,
and user-defined category rules using SQLite.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd

DEFAULT_DB_PATH = Path("data/spendscope.db")


def get_db_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create or return an SQLite connection with WAL mode and row factory."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    return conn


@contextlib.contextmanager
def db_session(db_path: Path = DEFAULT_DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that ensures SQLite connections are properly closed on exit."""
    conn = get_db_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initialize database tables and schema indexes."""
    with db_session(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                session_name TEXT NOT NULL,
                file_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_rows INTEGER DEFAULT 0,
                total_income REAL DEFAULT 0.0,
                total_expense REAL DEFAULT 0.0,
                net_savings REAL DEFAULT 0.0,
                date_min TEXT,
                date_max TEXT,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                date TEXT NOT NULL,
                description TEXT NOT NULL,
                raw_description TEXT,
                merchant TEXT,
                category TEXT,
                amount REAL NOT NULL,
                transaction_type TEXT NOT NULL,
                net_amount REAL NOT NULL,
                year_month TEXT,
                is_anomaly INTEGER DEFAULT 0,
                anomaly_score REAL DEFAULT 0.0,
                anomaly_severity TEXT DEFAULT 'Normal',
                anomaly_explanation TEXT,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS custom_rules (
                rule_id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                pattern TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS budgets (
                category TEXT PRIMARY KEY,
                monthly_limit REAL NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_txn_session ON transactions(session_id);
            CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(date);
            CREATE INDEX IF NOT EXISTS idx_txn_category ON transactions(category);
            CREATE INDEX IF NOT EXISTS idx_txn_merchant ON transactions(merchant);
            """
        )


def save_session(
    df: pd.DataFrame,
    session_name: str,
    file_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> str:
    """
    Persist an analysis session and its associated transactions to SQLite.

    Returns:
        The generated session_id.
    """
    init_database(db_path)
    session_id = f"SES-{uuid.uuid4().hex[:8].upper()}"

    income_mask = df["transaction_type"] == "income"
    expense_mask = df["transaction_type"] == "expense"
    total_inc = float(df.loc[income_mask, "amount"].sum())
    total_exp = float(df.loc[expense_mask, "amount"].sum())
    net_sav = total_inc - total_exp

    date_min = str(df["date"].min())[:10] if not df.empty else None
    date_max = str(df["date"].max())[:10] if not df.empty else None

    with db_session(db_path) as conn:
        # Insert session record
        conn.execute(
            """
            INSERT INTO sessions (
                session_id, session_name, file_name, created_at,
                total_rows, total_income, total_expense, net_savings,
                date_min, date_max, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                session_name,
                file_name or "Unknown",
                datetime.utcnow().isoformat(),
                len(df),
                total_inc,
                total_exp,
                net_sav,
                date_min,
                date_max,
                json.dumps(metadata or {}),
            ),
        )

        # Bulk insert transactions
        records = []
        for _, row in df.iterrows():
            records.append(
                (
                    str(row["transaction_id"]),
                    session_id,
                    str(row["date"])[:10],
                    str(row["description"]),
                    str(row.get("raw_description", "")),
                    str(row.get("merchant", "Unknown Merchant")),
                    str(row.get("category", "Other / Miscellaneous")),
                    float(row["amount"]),
                    str(row["transaction_type"]),
                    float(row["net_amount"]),
                    str(row.get("year_month", "")),
                    int(row.get("is_anomaly", False)),
                    float(row.get("anomaly_score", 0.0)),
                    str(row.get("anomaly_severity", "Normal")),
                    str(row.get("anomaly_explanation", "")),
                )
            )

        conn.executemany(
            """
            INSERT INTO transactions (
                transaction_id, session_id, date, description, raw_description,
                merchant, category, amount, transaction_type, net_amount,
                year_month, is_anomaly, anomaly_score, anomaly_severity,
                anomaly_explanation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            records,
        )

    return session_id


def list_sessions(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve metadata for all saved analysis sessions."""
    init_database(db_path)
    with db_session(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT session_id, session_name, file_name, created_at,
                   total_rows, total_income, total_expense, net_savings,
                   date_min, date_max
            FROM sessions
            ORDER BY created_at DESC
            """
        )
        return [dict(row) for row in cursor.fetchall()]


def load_session(session_id: str, db_path: Path = DEFAULT_DB_PATH) -> Optional[pd.DataFrame]:
    """Load all transactions for a specific session as a DataFrame."""
    init_database(db_path)
    with db_session(db_path) as conn:
        query = "SELECT * FROM transactions WHERE session_id = ? ORDER BY date ASC"
        df = pd.read_sql_query(query, conn, params=(session_id,))
        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df["is_anomaly"] = df["is_anomaly"].astype(bool)
        return df


def delete_session(session_id: str, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Delete a session and all its stored transactions."""
    init_database(db_path)
    with db_session(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM transactions WHERE session_id = ?", (session_id,))
        return cursor.rowcount > 0


# --- Custom Category Rules CRUD ---

def add_custom_rule(category: str, pattern: str, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Add a user-defined category matching pattern."""
    init_database(db_path)
    with db_session(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO custom_rules (category, pattern) VALUES (?, ?)",
            (category.strip(), pattern.strip().lower()),
        )
        return cursor.lastrowid or 0


def get_custom_rules(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all user-defined category rules."""
    init_database(db_path)
    with db_session(db_path) as conn:
        cursor = conn.execute("SELECT rule_id, category, pattern, created_at FROM custom_rules ORDER BY rule_id ASC")
        return [dict(row) for row in cursor.fetchall()]


def delete_custom_rule(rule_id: int, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Remove a user-defined category rule by ID."""
    init_database(db_path)
    with db_session(db_path) as conn:
        cursor = conn.execute("DELETE FROM custom_rules WHERE rule_id = ?", (rule_id,))
        return cursor.rowcount > 0


# --- Monthly Budget Targets CRUD ---

def set_category_budget(category: str, monthly_limit: float, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Upsert a monthly budget cap for a spending category."""
    init_database(db_path)
    with db_session(db_path) as conn:
        conn.execute(
            """
            INSERT INTO budgets (category, monthly_limit, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(category) DO UPDATE SET
                monthly_limit = excluded.monthly_limit,
                updated_at = CURRENT_TIMESTAMP
            """,
            (category.strip(), float(monthly_limit)),
        )


def get_category_budgets(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, float]:
    """Retrieve map of category -> monthly budget limit."""
    init_database(db_path)
    with db_session(db_path) as conn:
        cursor = conn.execute("SELECT category, monthly_limit FROM budgets")
        return {row["category"]: float(row["monthly_limit"]) for row in cursor.fetchall()}


def delete_category_budget(category: str, db_path: Path = DEFAULT_DB_PATH) -> bool:
    """Delete a category budget cap."""
    init_database(db_path)
    with db_session(db_path) as conn:
        cursor = conn.execute("DELETE FROM budgets WHERE category = ?", (category.strip(),))
        return cursor.rowcount > 0

