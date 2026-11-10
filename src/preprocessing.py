"""
SpendScope - Preprocessing Module.

Cleanses raw ingested financial data, parses heterogeneous date formats,
normalizes amounts, determines income vs expense transactions, engineers
temporal features, and ensures data integrity.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ingestion import (
    CANONICAL_AMOUNT,
    CANONICAL_CATEGORY,
    CANONICAL_CREDIT,
    CANONICAL_DATE,
    CANONICAL_DEBIT,
    CANONICAL_DESCRIPTION,
    CANONICAL_TYPE,
)


@dataclass
class PreprocessingReport:
    """Summary of data cleansing and transformations performed."""
    initial_rows: int
    final_rows: int
    dropped_rows: int
    duplicate_rows: int
    income_count: int
    expense_count: int
    total_income: float
    total_expense: float
    date_min: Optional[str]
    date_max: Optional[str]
    currency_symbol: str = "$"

    @property
    def net_savings(self) -> float:
        return self.total_income - self.total_expense

    def summary(self) -> str:
        sym = self.currency_symbol
        return (
            f"Preprocessed {self.final_rows} valid records "
            f"({self.dropped_rows} dropped, {self.duplicate_rows} duplicates flagged). "
            f"Income: {self.income_count} ({sym}{self.total_income:,.2f}) | "
            f"Expenses: {self.expense_count} ({sym}{self.total_expense:,.2f}) | "
            f"Net: {sym}{self.net_savings:,.2f}."
        )


COMMON_CURRENCY_SYMBOLS = [
    ("$", r"\$"),
    ("€", r"€"),
    ("£", r"£"),
    ("₹", r"₹"),
    ("¥", r"¥"),
    ("₩", r"₩"),
    ("CHF", r"\bCHF\b"),
    ("kr", r"\bkr\b"),
    ("R$", r"R\$"),
    ("zł", r"zł"),
    ("AED", r"\bAED\b"),
    ("SAR", r"\bSAR\b"),
    ("CAD", r"\bCAD\b|C\$"),
    ("AUD", r"\bAUD\b|A\$"),
]


def detect_currency_symbol(series: pd.Series) -> str:
    """
    Detect the predominant currency symbol from a series of raw monetary strings.
    Defaults to '$' if no explicit symbol is found.
    """
    counts: Dict[str, int] = {sym: 0 for sym, _ in COMMON_CURRENCY_SYMBOLS}
    non_null_samples = series.dropna().astype(str).head(200)

    for val in non_null_samples:
        for sym, pattern in COMMON_CURRENCY_SYMBOLS:
            if re.search(pattern, val, re.IGNORECASE):
                counts[sym] += 1

    max_sym = max(counts, key=counts.get)
    return max_sym if counts[max_sym] > 0 else "$"


def format_currency(amount: float, currency_symbol: str = "$") -> str:
    """
    Format numeric monetary value with comma grouping and dynamic currency symbol.
    """
    if pd.isna(amount):
        return f"{currency_symbol}0.00"
    if amount < 0:
        return f"-{currency_symbol}{abs(amount):,.2f}"
    return f"{currency_symbol}{amount:,.2f}"



def clean_currency_string(val: object) -> Optional[float]:
    """
    Parse messy monetary string representations into float.
    Handles currency symbols, commas, accounting brackets `(123.45)`,
    and trailing minus `123.45-`.
    """
    if val is None or pd.isna(val):
        return None

    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip()
    if not s:
        return None

    # Check for accounting negative format: (123.45)
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1].strip()
    elif s.endswith("-"):
        is_negative = True
        s = s[:-1].strip()
    elif s.startswith("-"):
        is_negative = True
        s = s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()

    # Strip currency symbols and thousands separators
    # Remove $, €, £, ₹, ¥ and commas
    cleaned = re.sub(r"[^\d.]", "", s)
    if not cleaned:
        return None

    try:
        num = float(cleaned)
        return -num if is_negative else num
    except ValueError:
        return None


def clean_description(val: object) -> str:
    """Standardize description strings: strip whitespace, reduce multi-spaces."""
    if val is None or pd.isna(val):
        return "Unknown Description"
    s = str(val).strip()
    # Replace multiple spaces/tabs with single space
    s = re.sub(r"\s+", " ", s)
    return s if s else "Unknown Description"


def parse_dates_robustly(series: pd.Series) -> pd.Series:
    """
    Parse a series of date strings into datetime64 with fallback strategies.
    Supports ISO (YYYY-MM-DD), US (MM/DD/YYYY), EU (DD/MM/YYYY), and timestamp formats.
    """
    # First attempt: standard pandas parsing with dayfirst=False
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed


def preprocess_transactions(
    raw_df: pd.DataFrame,
    mapping: Dict[str, str],
    drop_duplicates: bool = False,
    default_expense_if_positive: bool = False,
    override_currency: Optional[str] = None,
) -> Tuple[pd.DataFrame, PreprocessingReport]:
    """
    Transform raw ingested DataFrame into standard SpendScope canonical DataFrame.

    Canonical output schema:
        - transaction_id: str
        - date: pd.Timestamp
        - description: str
        - raw_description: str
        - amount: float (always positive)
        - transaction_type: str ("income" or "expense")
        - net_amount: float (+amount for income, -amount for expense)
        - currency_symbol: str
        - year: int
        - month: int
        - month_name: str
        - year_month: str (YYYY-MM)
        - day: int
        - day_name: str
        - day_of_week: int (0=Monday, 6=Sunday)
        - is_weekend: bool
        - quarter: int
        - is_duplicate: bool
        - raw_category: Optional[str] (if present in raw source)
    """
    initial_rows = len(raw_df)
    df = raw_df.copy()

    # Detect currency symbol from raw amount or debit/credit column
    detected_currency = "$"
    if override_currency:
        detected_currency = override_currency
    else:
        sample_currency_series = None
        if CANONICAL_AMOUNT in mapping and mapping[CANONICAL_AMOUNT] in df.columns:
            sample_currency_series = df[mapping[CANONICAL_AMOUNT]]
        elif CANONICAL_DEBIT in mapping and mapping[CANONICAL_DEBIT] in df.columns:
            sample_currency_series = df[mapping[CANONICAL_DEBIT]]
        elif CANONICAL_CREDIT in mapping and mapping[CANONICAL_CREDIT] in df.columns:
            sample_currency_series = df[mapping[CANONICAL_CREDIT]]

        if sample_currency_series is not None:
            detected_currency = detect_currency_symbol(sample_currency_series)

    # 1. Parse Dates
    date_col = mapping.get(CANONICAL_DATE)
    if not date_col or date_col not in df.columns:
        raise ValueError("Cannot preprocess dataset: Date column not mapped.")

    df["_parsed_date"] = parse_dates_robustly(df[date_col])

    # 2. Parse Description
    desc_col = mapping.get(CANONICAL_DESCRIPTION)
    if not desc_col or desc_col not in df.columns:
        raise ValueError("Cannot preprocess dataset: Description column not mapped.")

    df["raw_description"] = df[desc_col].astype(str)
    df["description"] = df[desc_col].apply(clean_description)

    # 3. Parse Amount and Transaction Type
    has_amount = CANONICAL_AMOUNT in mapping and mapping[CANONICAL_AMOUNT] in df.columns
    has_debit = CANONICAL_DEBIT in mapping and mapping[CANONICAL_DEBIT] in df.columns
    has_credit = CANONICAL_CREDIT in mapping and mapping[CANONICAL_CREDIT] in df.columns
    has_type = CANONICAL_TYPE in mapping and mapping[CANONICAL_TYPE] in df.columns

    parsed_amounts: List[Optional[float]] = []
    parsed_types: List[str] = []

    if has_debit or has_credit:
        # Split debit/credit columns
        debit_vals = df[mapping[CANONICAL_DEBIT]].apply(clean_currency_string) if has_debit else pd.Series([None] * len(df))
        credit_vals = df[mapping[CANONICAL_CREDIT]].apply(clean_currency_string) if has_credit else pd.Series([None] * len(df))

        for d, c in zip(debit_vals, credit_vals):
            d_val = abs(d) if (d is not None and not pd.isna(d) and d != 0) else None
            c_val = abs(c) if (c is not None and not pd.isna(c) and c != 0) else None

            if c_val is not None:
                parsed_amounts.append(c_val)
                parsed_types.append("income")
            elif d_val is not None:
                parsed_amounts.append(d_val)
                parsed_types.append("expense")
            else:
                parsed_amounts.append(None)
                parsed_types.append("expense")

    elif has_amount:
        raw_amounts = df[mapping[CANONICAL_AMOUNT]].apply(clean_currency_string)
        type_series = df[mapping[CANONICAL_TYPE]].astype(str).str.lower().str.strip() if has_type else None

        income_type_keywords = {"income", "credit", "cr", "deposit", "in", "incoming"}
        expense_type_keywords = {"expense", "debit", "dr", "withdrawal", "out", "outgoing", "payment"}

        for idx, amt in enumerate(raw_amounts):
            if amt is None or pd.isna(amt):
                parsed_amounts.append(None)
                parsed_types.append("expense")
                continue

            # Check explicit type column if available
            explicit_type = None
            if type_series is not None:
                t_val = type_series.iloc[idx]
                if t_val in income_type_keywords:
                    explicit_type = "income"
                elif t_val in expense_type_keywords:
                    explicit_type = "expense"

            if explicit_type is not None:
                parsed_amounts.append(abs(amt))
                parsed_types.append(explicit_type)
            else:
                # Deduce from sign
                if amt < 0:
                    parsed_amounts.append(abs(amt))
                    parsed_types.append("expense")
                elif amt > 0:
                    # Positive amount
                    if default_expense_if_positive:
                        parsed_amounts.append(amt)
                        parsed_types.append("expense")
                    else:
                        parsed_amounts.append(amt)
                        parsed_types.append("income")
                else:
                    parsed_amounts.append(0.0)
                    parsed_types.append("expense")
    else:
        raise ValueError("Cannot preprocess dataset: Financial value (amount or debit/credit) not found.")

    df["amount"] = parsed_amounts
    df["transaction_type"] = parsed_types

    # Drop records with invalid date or invalid amount
    valid_mask = df["_parsed_date"].notna() & df["amount"].notna()
    dropped_rows = initial_rows - int(valid_mask.sum())
    clean_df = df[valid_mask].copy()

    # Assign standardized columns
    clean_df["date"] = clean_df["_parsed_date"]
    clean_df["amount"] = clean_df["amount"].astype(float)
    clean_df["net_amount"] = np.where(
        clean_df["transaction_type"] == "income",
        clean_df["amount"],
        -clean_df["amount"],
    )
    clean_df["currency_symbol"] = detected_currency

    # 4. Feature Engineering: Temporal dimensions
    clean_df["year"] = clean_df["date"].dt.year
    clean_df["month"] = clean_df["date"].dt.month
    clean_df["month_name"] = clean_df["date"].dt.strftime("%b")
    clean_df["year_month"] = clean_df["date"].dt.strftime("%Y-%m")
    clean_df["day"] = clean_df["date"].dt.day
    clean_df["day_name"] = clean_df["date"].dt.day_name()
    clean_df["day_of_week"] = clean_df["date"].dt.dayofweek
    clean_df["is_weekend"] = clean_df["day_of_week"].isin([5, 6])
    clean_df["quarter"] = clean_df["date"].dt.quarter

    # Preserve optional raw category
    if CANONICAL_CATEGORY in mapping and mapping[CANONICAL_CATEGORY] in clean_df.columns:
        clean_df["raw_category"] = clean_df[mapping[CANONICAL_CATEGORY]].astype(str)
    else:
        clean_df["raw_category"] = None

    # 5. Duplicate Detection
    duplicate_mask = clean_df.duplicated(
        subset=["date", "description", "amount", "transaction_type"],
        keep="first",
    )
    clean_df["is_duplicate"] = duplicate_mask
    duplicate_count = int(duplicate_mask.sum())

    if drop_duplicates and duplicate_count > 0:
        clean_df = clean_df[~duplicate_mask].copy()

    # 6. Generate Unique Transaction IDs
    clean_df["transaction_id"] = [
        f"TXN-{uuid.uuid4().hex[:8].upper()}" for _ in range(len(clean_df))
    ]

    # Sort chronologically
    clean_df = clean_df.sort_values(by=["date", "transaction_id"]).reset_index(drop=True)

    # Build canonical columns list
    canonical_columns = [
        "transaction_id",
        "date",
        "description",
        "raw_description",
        "amount",
        "transaction_type",
        "net_amount",
        "currency_symbol",
        "year",
        "month",
        "month_name",
        "year_month",
        "day",
        "day_name",
        "day_of_week",
        "is_weekend",
        "quarter",
        "is_duplicate",
        "raw_category",
    ]

    final_df = clean_df[canonical_columns].copy()

    # 7. Preprocessing Summary Report
    final_rows = len(final_df)
    income_subset = final_df[final_df["transaction_type"] == "income"]
    expense_subset = final_df[final_df["transaction_type"] == "expense"]

    report = PreprocessingReport(
        initial_rows=initial_rows,
        final_rows=final_rows,
        dropped_rows=dropped_rows,
        duplicate_rows=duplicate_count,
        income_count=len(income_subset),
        expense_count=len(expense_subset),
        total_income=float(income_subset["amount"].sum()),
        total_expense=float(expense_subset["amount"].sum()),
        date_min=final_df["date"].min().strftime("%Y-%m-%d") if final_rows > 0 else None,
        date_max=final_df["date"].max().strftime("%Y-%m-%d") if final_rows > 0 else None,
        currency_symbol=detected_currency,
    )

    return final_df, report
