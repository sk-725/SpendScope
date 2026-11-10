"""
SpendScope - Categorization Module.

Applies pattern-matching rules, keyword dictionaries, and regex heuristics
to automatically categorize transactions and extract normalized merchant names
from unstructured bank description strings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern, Tuple

import pandas as pd


@dataclass
class CategoryRule:
    """Represents a rule targeting specific keywords/patterns for a category."""
    category: str
    patterns: List[str]
    compiled_patterns: List[Pattern] = field(init=False)
    default_merchant: Optional[str] = None

    def __post_init__(self):
        self.compiled_patterns = [
            re.compile(r"\b" + re.escape(p).replace(r"\ ", r"\s+") + r"\b", re.IGNORECASE)
            for p in self.patterns
        ]


# Canonical SpendScope Categories with colors and icon representations
CATEGORY_METADATA = {
    "Income / Salary": {"color": "#10B981", "icon": "💰"},
    "Housing & Rent": {"color": "#6366F1", "icon": "🏠"},
    "Utilities & Bills": {"color": "#F59E0B", "icon": "⚡"},
    "Groceries & Supermarkets": {"color": "#3B82F6", "icon": "🛒"},
    "Dining & Food Delivery": {"color": "#EC4899", "icon": "🍔"},
    "Transportation & Fuel": {"color": "#14B8A6", "icon": "🚗"},
    "Entertainment & Streaming": {"color": "#8B5CF6", "icon": "🎬"},
    "Shopping & Retail": {"color": "#F97316", "icon": "🛍️"},
    "Health & Fitness": {"color": "#06B6D4", "icon": "💊"},
    "Travel & Lodging": {"color": "#EAB308", "icon": "✈️"},
    "Financial & Fees": {"color": "#64748B", "icon": "🏦"},
    "Other / Miscellaneous": {"color": "#94A3B8", "icon": "📦"},
}

DEFAULT_CATEGORY = "Other / Miscellaneous"

# Standard rule definitions
DEFAULT_RULES: List[CategoryRule] = [
    CategoryRule(
        category="Income / Salary",
        patterns=[
            "salary", "payroll", "direct dep", "direct deposit", "paycheck",
            "dividend", "freelance", "stipend", "tax refund", "irs federal",
            "bonus acme", "performance bonus", "client payment",
        ],
    ),
    CategoryRule(
        category="Housing & Rent",
        patterns=[
            "monthly rent", "apartments", "apartment", "mortgage", "hoa fee",
            "lease payment", "property management", "realty",
        ],
    ),
    CategoryRule(
        category="Utilities & Bills",
        patterns=[
            "conedison", "electric & gas", "electric utility", "power & light",
            "comcast", "xfinity", "broadband", "internet", "at&t", "verizon",
            "t-mobile", "water utility", "sewage", "trash collection",
        ],
    ),
    CategoryRule(
        category="Groceries & Supermarkets",
        patterns=[
            "trader joe", "whole foods", "costco", "kroger", "safeway",
            "aldi", "groceries", "supermarket", "market basket", "h-mart",
            "wegmans", "heb", "publix",
        ],
    ),
    CategoryRule(
        category="Dining & Food Delivery",
        patterns=[
            "starbucks", "chipotle", "shake shack", "mcdonald", "subway sandwich",
            "doordash", "uber eats", "grubhub", "burger", "pizza", "cafe",
            "coffee", "bakery", "bistro", "diner", "restaurant",
        ],
    ),
    CategoryRule(
        category="Transportation & Fuel",
        patterns=[
            "shell oil", "chevron", "bp gas", "exxon", "fuel station",
            "gas station", "uber", "lyft", "mta", "metrocard", "subway token",
            "transit", "parking", "toll", "ezpass", "train ticket",
        ],
    ),
    CategoryRule(
        category="Entertainment & Streaming",
        patterns=[
            "netflix", "spotify", "hulu", "disney", "apple.com/bill",
            "icloud storage", "steam games", "playstation", "cinema", "amc",
            "youtube premium", "hbo", "paramount", "audible",
        ],
    ),
    CategoryRule(
        category="Shopping & Retail",
        patterns=[
            "amazon", "target", "walmart", "best buy", "apple store",
            "tiffany & co", "zara", "h&m", "nike", "bookstore",
            "clothing", "hardware", "home goods",
        ],
    ),
    CategoryRule(
        category="Health & Fitness",
        patterns=[
            "planet fitness", "gym membership", "cvs pharmacy", "cvs",
            "walgreens", "rite aid", "pharmacy", "dental", "clinic",
            "hospital", "doctor", "health",
        ],
    ),
    CategoryRule(
        category="Travel & Lodging",
        patterns=[
            "lufthansa", "airline", "airlines", "delta", "united air",
            "american air", "flight", "hotel", "airbnb", "marriott",
            "hilton", "booking.com", "expedia",
        ],
    ),
    CategoryRule(
        category="Financial & Fees",
        patterns=[
            "atm cash", "atm withdrawal", "bank fee", "wire fee", "overdraft",
            "interest charge", "vanguard", "fidelity", "charles schwab",
        ],
    ),
]

# Common merchant cleaning dictionary
KNOWN_MERCHANT_PATTERNS = [
    (r"(?i)starbucks.*", "Starbucks"),
    (r"(?i)trader joe.*", "Trader Joe's"),
    (r"(?i)whole foods.*", "Whole Foods Market"),
    (r"(?i)netflix.*", "Netflix"),
    (r"(?i)spotify.*", "Spotify"),
    (r"(?i)icloud.*", "Apple iCloud"),
    (r"(?i)apple store.*", "Apple Store"),
    (r"(?i)shell oil.*", "Shell"),
    (r"(?i)chevron.*", "Chevron"),
    (r"(?i)conedison.*", "ConEdison"),
    (r"(?i)comcast.*|xfinity.*", "Comcast Xfinity"),
    (r"(?i)planet fitness.*", "Planet Fitness"),
    (r"(?i)chipotle.*", "Chipotle"),
    (r"(?i)shake shack.*", "Shake Shack"),
    (r"(?i)subway sandwich.*", "Subway"),
    (r"(?i)amazon.*", "Amazon"),
    (r"(?i)target.*", "Target"),
    (r"(?i)costco.*", "Costco"),
    (r"(?i)best buy.*", "Best Buy"),
    (r"(?i)cvs.*", "CVS Pharmacy"),
    (r"(?i)walgreens.*", "Walgreens"),
    (r"(?i)doordash.*", "DoorDash"),
    (r"(?i)uber eats.*", "Uber Eats"),
    (r"(?i)uber\s*\*?.*", "Uber"),
    (r"(?i)lyft.*", "Lyft"),
    (r"(?i)mta.*", "MTA Transit"),
    (r"(?i)tiffany & co.*", "Tiffany & Co"),
    (r"(?i)lufthansa.*", "Lufthansa"),
    (r"(?i)zara.*", "Zara"),
    (r"(?i)techcorp.*", "TechCorp"),
    (r"(?i)highland apartments.*", "Highland Apartments"),
    (r"(?i)irs federal.*", "IRS Tax Refund"),
    (r"(?i)late night atm.*", "ATM Cash Withdrawal"),
    (r"(?i)emergency plumbing.*", "Emergency Plumbing"),
    (r"(?i)freelance.*", "Freelance Client"),
    (r"(?i)stock dividend.*", "Stock Dividend"),
    (r"(?i)performance bonus.*", "Acme Corp Bonus"),
]


class TransactionCategorizer:
    """
    Intelligent transaction categorization engine.
    Applies regex heuristics, merchant normalization, and confidence estimation.
    """

    def __init__(self, custom_rules: Optional[List[CategoryRule]] = None):
        self.rules = (custom_rules or []) + DEFAULT_RULES
        self._merchant_cleaners = [
            (re.compile(pattern), name) for pattern, name in KNOWN_MERCHANT_PATTERNS
        ]

    @classmethod
    def from_database(cls, db_path: Optional[Any] = None) -> "TransactionCategorizer":
        """Instantiate categorizer incorporating custom rules saved in SQLite."""
        try:
            from src.database import get_custom_rules
            saved_rules = get_custom_rules(db_path) if db_path else get_custom_rules()
            category_patterns: Dict[str, List[str]] = {}
            for r in saved_rules:
                cat = r["category"]
                pat = r["pattern"]
                category_patterns.setdefault(cat, []).append(pat)

            custom_category_rules = [
                CategoryRule(category=cat, patterns=pats)
                for cat, pats in category_patterns.items()
            ]
            return cls(custom_rules=custom_category_rules)
        except Exception:
            return cls()

    def add_runtime_rule(self, category: str, pattern: str) -> None:
        """Add a dynamic custom rule at runtime with top matching priority."""
        new_rule = CategoryRule(category=category, patterns=[pattern])
        self.rules.insert(0, new_rule)

    def extract_merchant(self, description: str) -> str:
        """Extract a clean, normalized merchant or brand name from description."""
        if not description:
            return "Unknown Merchant"

        clean_desc = description.strip()
        for regex, merchant_name in self._merchant_cleaners:
            if regex.search(clean_desc):
                return merchant_name

        # Fallback: strip order numbers, store IDs, state codes
        simplified = re.sub(r"#[0-9]+", "", clean_desc)
        simplified = re.sub(r"\*[0-9A-Z]+", "", simplified)
        simplified = re.sub(r"\s+", " ", simplified).strip()
        return simplified if simplified else "Unknown Merchant"

    def categorize_description(
        self,
        description: str,
        transaction_type: str = "expense",
        raw_category: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Classify description into a standardized category.

        Returns:
            Tuple of (category_name, confidence_score between 0.0 and 1.0)
        """
        if not description or description.lower() == "unknown description":
            return DEFAULT_CATEGORY, 0.1

        # Income heuristic override if transaction type is income
        if transaction_type == "income":
            return "Income / Salary", 0.95

        desc_lower = description.lower()

        # Match against predefined category rules
        for rule in self.rules:
            # Income rule is skipped if transaction_type is strictly expense
            if rule.category == "Income / Salary" and transaction_type == "expense":
                continue

            for pattern_re in rule.compiled_patterns:
                if pattern_re.search(desc_lower):
                    return rule.category, 0.90

        # If raw category was provided in the file, try fuzzy mapping
        if raw_category and pd.notna(raw_category):
            raw_cat_lower = str(raw_category).lower()
            for cat_name in CATEGORY_METADATA.keys():
                if cat_name.lower() in raw_cat_lower or raw_cat_lower in cat_name.lower():
                    return cat_name, 0.75

        return DEFAULT_CATEGORY, 0.30

    def categorize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized/batch categorization for canonical SpendScope DataFrames.
        Appends `category`, `merchant`, `category_confidence`, and visual indicators.
        """
        df_out = df.copy()

        categories: List[str] = []
        confidences: List[float] = []
        merchants: List[str] = []

        for _, row in df_out.iterrows():
            desc = row.get("description", "")
            t_type = row.get("transaction_type", "expense")
            raw_cat = row.get("raw_category", None)

            cat, conf = self.categorize_description(desc, t_type, raw_cat)
            merchant = self.extract_merchant(desc)

            categories.append(cat)
            confidences.append(conf)
            merchants.append(merchant)

        df_out["category"] = categories
        df_out["category_confidence"] = confidences
        df_out["merchant"] = merchants

        # Add category color and emoji icon
        df_out["category_icon"] = df_out["category"].map(
            lambda c: CATEGORY_METADATA.get(c, {}).get("icon", "📦")
        )
        df_out["category_color"] = df_out["category"].map(
            lambda c: CATEGORY_METADATA.get(c, {}).get("color", "#94A3B8")
        )

        return df_out
