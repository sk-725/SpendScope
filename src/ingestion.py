"""
SpendScope - Data Ingestion and Validation Module.

Handles loading, validating, and standardizing CSV financial files from various
bank export formats, handling delimiters, encodings, and heterogeneous schemas.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Optional, TextIO, Tuple, Union

import pandas as pd


# Canonical column keys expected by SpendScope downstream pipeline
CANONICAL_DATE = "date"
CANONICAL_DESCRIPTION = "description"
CANONICAL_AMOUNT = "amount"
CANONICAL_TYPE = "type"
CANONICAL_DEBIT = "debit"
CANONICAL_CREDIT = "credit"
CANONICAL_CATEGORY = "category"

# Common synonyms across major banks and personal finance software
SYNONYMS: Dict[str, List[str]] = {
    CANONICAL_DATE: [
        "date",
        "txn_date",
        "transaction_date",
        "transaction date",
        "trans date",
        "post_date",
        "post date",
        "posted date",
        "booking date",
        "value date",
        "time",
        "timestamp",
    ],
    CANONICAL_DESCRIPTION: [
        "description",
        "desc",
        "merchant",
        "payee",
        "details",
        "narrative",
        "memo",
        "name",
        "transaction description",
        "particulars",
        "transaction details",
    ],
    CANONICAL_AMOUNT: [
        "amount",
        "total",
        "value",
        "net",
        "txn_amount",
        "transaction amount",
        "sum",
    ],
    CANONICAL_DEBIT: [
        "debit",
        "withdrawal",
        "withdrawals",
        "spent",
        "charge",
        "money out",
        "outgoing",
        "debit amount",
    ],
    CANONICAL_CREDIT: [
        "credit",
        "deposit",
        "deposits",
        "received",
        "money in",
        "incoming",
        "credit amount",
    ],
    CANONICAL_TYPE: [
        "type",
        "transaction type",
        "trans type",
        "cr/dr",
        "dr/cr",
        "direction",
    ],
    CANONICAL_CATEGORY: [
        "category",
        "tag",
        "tags",
        "classification",
        "spending category",
    ],
}


@dataclass
class ValidationResult:
    """Encapsulates the outcome of schema validation and ingestion."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    detected_columns: Dict[str, str] = field(default_factory=dict)
    total_raw_rows: int = 0
    preview_rows: int = 0

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

    def summary(self) -> str:
        """User-friendly summary of validation status."""
        if self.is_valid:
            warn_str = f" with {len(self.warnings)} warning(s)" if self.has_warnings else ""
            return f"Validation Passed: {self.total_raw_rows} rows ingested{warn_str}."
        return f"Validation Failed: {'; '.join(self.errors)}"


def detect_columns(columns: List[str]) -> Dict[str, str]:
    """
    Automatically map raw CSV column headers to canonical SpendScope keys.

    Args:
        columns: List of raw column headers.

    Returns:
        Dict mapping canonical keys (e.g. 'date', 'amount') to raw column names.
    """
    normalized_cols = {col: col.strip().lower().replace("_", " ") for col in columns}
    mapping: Dict[str, str] = {}

    for canonical, candidates in SYNONYMS.items():
        for raw_col, norm_col in normalized_cols.items():
            norm_col_clean = norm_col.replace("-", " ")
            # Direct match against candidates
            if norm_col in candidates or norm_col_clean in candidates:
                mapping[canonical] = raw_col
                break
            # Substring match if no direct match found yet
            if canonical not in mapping:
                for candidate in candidates:
                    if candidate in norm_col:
                        mapping[canonical] = raw_col
                        break

    return mapping


def detect_delimiter(sample_text: str) -> str:
    """Detect delimiter (comma, semicolon, tab, pipe) from a sample text snippet."""
    first_lines = "\n".join(sample_text.splitlines()[:5])
    counts = {
        ",": first_lines.count(","),
        ";": first_lines.count(";"),
        "\t": first_lines.count("\t"),
        "|": first_lines.count("|"),
    }
    # Pick delimiter with highest occurrence
    best_delim = max(counts, key=counts.get)
    return best_delim if counts[best_delim] > 0 else ","


def validate_raw_dataframe(df: pd.DataFrame, mapping: Dict[str, str]) -> ValidationResult:
    """
    Inspect raw DataFrame schema against SpendScope requirements.

    Args:
        df: Loaded raw Pandas DataFrame.
        mapping: Mapping of canonical keys to raw columns.

    Returns:
        ValidationResult with validation flags, errors, and warnings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    total_rows = len(df)
    if total_rows == 0:
        errors.append("CSV file contains no data rows (empty dataset).")
        return ValidationResult(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            detected_columns=mapping,
            total_raw_rows=0,
        )

    # Required: Date
    if CANONICAL_DATE not in mapping:
        errors.append(
            "Missing required Date column. Expected headers such as: Date, Transaction Date, Post Date."
        )

    # Required: Description
    if CANONICAL_DESCRIPTION not in mapping:
        errors.append(
            "Missing required Description column. Expected headers such as: Description, Merchant, Details, Payee."
        )

    # Required: Amount (either unified 'amount' OR split 'debit'/'credit')
    has_amount = CANONICAL_AMOUNT in mapping
    has_split = CANONICAL_DEBIT in mapping or CANONICAL_CREDIT in mapping
    if not (has_amount or has_split):
        errors.append(
            "Missing financial value column. Must provide either 'Amount' or 'Debit'/'Credit' columns."
        )

    # Check for empty columns
    for canonical_name, raw_name in mapping.items():
        if df[raw_name].isna().all():
            warnings.append(f"Mapped column '{raw_name}' ({canonical_name}) contains 100% missing values.")

    # Check if more than 30% of critical columns are blank
    if CANONICAL_DATE in mapping and df[mapping[CANONICAL_DATE]].isna().mean() > 0.3:
        warnings.append(
            f"Over 30% of rows in '{mapping[CANONICAL_DATE]}' are missing dates and may be discarded."
        )

    if has_amount and df[mapping[CANONICAL_AMOUNT]].isna().mean() > 0.3:
        warnings.append(
            f"Over 30% of rows in '{mapping[CANONICAL_AMOUNT]}' are missing amounts and may be discarded."
        )

    is_valid = len(errors) == 0
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        detected_columns=mapping,
        total_raw_rows=total_rows,
        preview_rows=min(5, total_rows),
    )


def extract_pdf_statement_to_df(content_bytes: bytes) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Extract financial transactions from a PDF bank statement using pdfplumber.
    Attempts table extraction first, then falls back to regex line parsing.

    Returns:
        (DataFrame, None) on success, or (None, error_message) on failure.
    """
    try:
        import pdfplumber
    except ImportError:
        return None, "pdfplumber package is required for PDF statement extraction."

    try:
        all_rows: List[List[str]] = []
        headers: Optional[List[str]] = None

        with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
            if len(pdf.pages) == 0:
                return None, "PDF file contains no pages."

            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
                            if not any(cleaned_row):
                                continue

                            # Identify header row if not yet established
                            if headers is None:
                                mapped = detect_columns(cleaned_row)
                                if CANONICAL_DATE in mapped or CANONICAL_AMOUNT in mapped:
                                    headers = cleaned_row
                                    continue
                            all_rows.append(cleaned_row)

        if headers and all_rows:
            max_cols = len(headers)
            normalized_rows = [
                row[:max_cols] + [""] * (max_cols - len(row)) if len(row) < max_cols else row[:max_cols]
                for row in all_rows
            ]
            df = pd.DataFrame(normalized_rows, columns=headers)
            return df, None

        # Fallback: line-by-line regex parsing
        import re
        text_rows: List[Dict[str, str]] = []
        date_pattern = re.compile(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})")
        amount_pattern = re.compile(r"([+-]?\$?[\d,]+\.\d{2})")

        with pdfplumber.open(io.BytesIO(content_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                for line in text.splitlines():
                    d_match = date_pattern.search(line)
                    a_matches = amount_pattern.findall(line)
                    if d_match and a_matches:
                        d_str = d_match.group(1)
                        amt_str = a_matches[-1]
                        start_pos = d_match.end()
                        end_pos = line.rfind(amt_str)
                        desc_str = (
                            line[start_pos:end_pos].strip()
                            if end_pos > start_pos
                            else line.replace(d_str, "").replace(amt_str, "").strip()
                        )
                        if desc_str and amt_str:
                            text_rows.append({"Date": d_str, "Description": desc_str, "Amount": amt_str})

        if text_rows:
            return pd.DataFrame(text_rows), None

        return None, "No structured transaction tables or recognized transaction patterns found in PDF."
    except Exception as e:
        return None, f"PDF extraction error: {str(e)}"


def read_csv_safely(
    source: Union[str, Path, TextIO, BinaryIO, io.StringIO, io.BytesIO],
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    file_name: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Read a CSV or PDF source with automatic fallback across encodings, delimiters, and file types.

    Returns:
        (DataFrame, None) on success, or (None, error_message) on failure.
    """
    # Extract sample bytes / string if possible to sniff delimiter or format
    content_bytes: Optional[bytes] = None
    source_name = file_name or (str(source) if isinstance(source, (str, Path)) else "")

    if isinstance(source, (str, Path)):
        try:
            with open(source, "rb") as f:
                content_bytes = f.read()
        except Exception as e:
            return None, f"Failed to open file path '{source}': {str(e)}"
    elif hasattr(source, "read"):
        current_pos = source.tell() if hasattr(source, "tell") else None
        read_val = source.read()
        if isinstance(read_val, str):
            content_bytes = read_val.encode("utf-8")
        else:
            content_bytes = read_val
        # Reset position if possible
        if hasattr(source, "seek") and current_pos is not None:
            source.seek(current_pos)

    if content_bytes is None or len(content_bytes.strip()) == 0:
        return None, "File is empty or cannot be read."

    # Detect PDF by extension or magic bytes (%PDF)
    if source_name.lower().endswith(".pdf") or content_bytes.startswith(b"%PDF"):
        return extract_pdf_statement_to_df(content_bytes)

    encodings_to_try = [encoding] if encoding else ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    # Try encodings and delimiters
    last_err: Optional[str] = None
    for enc in encodings_to_try:
        try:
            text = content_bytes.decode(enc)
            delim = delimiter if delimiter else detect_delimiter(text)
            df = pd.read_csv(
                io.StringIO(text),
                sep=delim,
                skipinitialspace=True,
                dtype=str,  # read as string initially to prevent eager truncation
            )
            return df, None
        except Exception as e:
            last_err = str(e)
            continue

    return None, f"Failed to parse CSV with supported encodings ({last_err})"


def load_and_validate_csv(
    source: Union[str, Path, TextIO, BinaryIO, io.StringIO, io.BytesIO],
    custom_mapping: Optional[Dict[str, str]] = None,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    file_name: Optional[str] = None,
) -> Tuple[Optional[pd.DataFrame], ValidationResult]:
    """
    High-level entry point to ingest and validate a financial CSV or PDF statement file.

    Args:
        source: File path or file-like object.
        custom_mapping: Optional explicit mapping overriding auto-detection.
        delimiter: Optional explicit delimiter.
        encoding: Optional explicit character encoding.
        file_name: Optional filename for extension format resolution.

    Returns:
        Tuple of (Raw DataFrame, ValidationResult).
    """
    df, err = read_csv_safely(source, delimiter=delimiter, encoding=encoding, file_name=file_name)
    if err or df is None:
        val_result = ValidationResult(
            is_valid=False,
            errors=[err or "Unknown error reading file."],
            total_raw_rows=0,
        )
        return None, val_result

    # Detect or apply column mapping
    auto_mapping = detect_columns(list(df.columns))
    if custom_mapping:
        auto_mapping.update(custom_mapping)

    val_result = validate_raw_dataframe(df, auto_mapping)
    return df, val_result
