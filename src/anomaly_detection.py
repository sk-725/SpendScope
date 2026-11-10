"""
SpendScope - Explainable Anomaly Intelligence Module.

Employs Scikit-learn's IsolationForest combined with domain-specific feature
engineering and multi-layered contextual baselines (category distributions,
merchant history, overall spend ranking, and temporal patterns) to flag
statistically unusual transactions with transparent, understandable explanations.

IMPORTANT: Anomalies indicate statistical deviation, NOT confirmed financial fraud.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalySummary:
    """Statistical summary of detected anomalies."""
    total_evaluated: int
    anomalies_detected: int
    anomaly_percentage: float
    total_anomalous_spend: float
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int


class ExpenseAnomalyDetector:
    """
    Explainable Isolation Forest anomaly detector for personal financial transactions.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.scaler = StandardScaler()

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive numeric signals for Isolation Forest:
        - log-transformed amount
        - amount-to-category-median ratio
        - day of week, day of month, weekend indicator
        """
        feat = pd.DataFrame(index=df.index)
        feat["log_amount"] = np.log1p(df["amount"].astype(float))
        feat["day_of_week"] = df["day_of_week"].astype(float)
        feat["is_weekend"] = df["is_weekend"].astype(float)
        feat["day"] = df["day"].astype(float)

        # Ratio to category median
        category_medians = df.groupby("category")["amount"].transform("median")
        feat["cat_median_ratio"] = (df["amount"] / (category_medians + 1.0)).fillna(1.0)

        return feat.fillna(0.0)

    def fit_predict(
        self,
        df: pd.DataFrame,
        currency_symbol: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, AnomalySummary]:
        """
        Train Isolation Forest and score each transaction with rich contextual baselines.
        Appends `is_anomaly`, `anomaly_score`, `anomaly_severity`, `anomaly_tag`,
        and `anomaly_explanation`.
        """
        sym = currency_symbol or (df["currency_symbol"].iloc[0] if "currency_symbol" in df.columns and not df.empty else "$")
        expense_mask = df["transaction_type"] == "expense"
        df_out = df.copy()

        # Initialize anomaly columns
        df_out["is_anomaly"] = False
        df_out["anomaly_score"] = 0.0
        df_out["anomaly_severity"] = "Normal"
        df_out["anomaly_tag"] = "Normal"
        df_out["anomaly_explanation"] = ""

        if expense_mask.sum() < 5:
            summary = AnomalySummary(
                total_evaluated=int(expense_mask.sum()),
                anomalies_detected=0,
                anomaly_percentage=0.0,
                total_anomalous_spend=0.0,
                high_severity_count=0,
                medium_severity_count=0,
                low_severity_count=0,
            )
            return df_out, summary

        expense_df = df[expense_mask].copy()
        X = self._engineer_features(expense_df)
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        preds = self.model.fit_predict(X_scaled)
        raw_scores = self.model.decision_function(X_scaled)

        # Normalize score so higher = more anomalous (0 to 1 scale)
        min_s, max_s = raw_scores.min(), raw_scores.max()
        normalized_scores = 1.0 - ((raw_scores - min_s) / (max_s - min_s + 1e-9))

        is_anom_bool = preds == -1
        expense_df["is_anomaly"] = is_anom_bool
        expense_df["anomaly_score"] = np.round(normalized_scores, 3)

        # Precompute contextual baselines across the dataset
        cat_medians = expense_df.groupby("category")["amount"].median().to_dict()
        cat_q75 = expense_df.groupby("category")["amount"].quantile(0.75).to_dict()
        merchant_counts = expense_df["merchant"].value_counts().to_dict()
        merchant_medians = expense_df.groupby("merchant")["amount"].median().to_dict()
        
        # Rank transactions by amount descending
        sorted_indices = expense_df["amount"].sort_values(ascending=False).index
        rank_map = {idx: rank + 1 for rank, idx in enumerate(sorted_indices)}
        total_expense_records = len(expense_df)

        explanations: List[str] = []
        severities: List[str] = []
        tags: List[str] = []

        for idx, row in expense_df.iterrows():
            if row["is_anomaly"]:
                cat = row["category"]
                amt = float(row["amount"])
                merch = row.get("merchant", "Unknown Merchant")
                med = float(cat_medians.get(cat, amt))
                q75 = float(cat_q75.get(cat, amt))
                ratio = amt / (med + 1e-9)
                rank = rank_map.get(idx, 0)
                m_count = merchant_counts.get(merch, 1)
                m_med = float(merchant_medians.get(merch, amt))
                day_name = row.get("day_name", "weekday")

                # Contextual tag and explanation formulation
                if ratio >= 4.0 or rank <= 3:
                    sev = "High"
                    if m_count == 1:
                        tag = "First-Time High-Ticket Outlier"
                        exp = (
                            f"Ranked #{rank} largest purchase across {total_expense_records} transactions ({sym}{amt:,.2f}). "
                            f"Exceeds typical {cat} median ({sym}{med:,.2f}) by {ratio:.1f}x. First recorded transaction with {merch}."
                        )
                    else:
                        tag = "Category Scale Deviation"
                        exp = (
                            f"Amount ({sym}{amt:,.2f}) is {ratio:.1f}x higher than typical {cat} median ({sym}{med:,.2f}) "
                            f"(75th percentile: {sym}{q75:,.2f}). Ranked #{rank} overall."
                        )
                elif ratio >= 2.0 or (m_count >= 2 and amt > m_med * 2.0):
                    sev = "Medium"
                    if m_count >= 2 and amt > m_med * 1.8:
                        tag = "Merchant Baseline Surge"
                        exp = (
                            f"Spend at {merch} ({sym}{amt:,.2f}) is {(amt / m_med):.1f}x higher than your historical "
                            f"median for this vendor ({sym}{m_med:,.2f})."
                        )
                    else:
                        tag = "Elevated Spend Outlier"
                        exp = (
                            f"Transaction ({sym}{amt:,.2f}) sits above the 75th percentile ({sym}{q75:,.2f}) "
                            f"for {cat}, representing an uncommon volume for {day_name}."
                        )
                else:
                    sev = "Low"
                    tag = "Temporal / Frequency Anomaly"
                    exp = (
                        f"Atypical combination of timing ({day_name}) and spend pattern ({sym}{amt:,.2f}) "
                        f"relative to standard {cat} transactions."
                    )
            else:
                sev = "Normal"
                tag = "Standard Transaction"
                exp = ""

            severities.append(sev)
            tags.append(tag)
            explanations.append(exp)

        expense_df["anomaly_severity"] = severities
        expense_df["anomaly_tag"] = tags
        expense_df["anomaly_explanation"] = explanations

        # Merge back
        for col in ["is_anomaly", "anomaly_score", "anomaly_severity", "anomaly_tag", "anomaly_explanation"]:
            df_out.loc[expense_mask, col] = expense_df[col]

        # Summary
        anom_subset = df_out[df_out["is_anomaly"]]
        summary = AnomalySummary(
            total_evaluated=int(expense_mask.sum()),
            anomalies_detected=len(anom_subset),
            anomaly_percentage=float(len(anom_subset) / max(1, expense_mask.sum()) * 100.0),
            total_anomalous_spend=float(anom_subset["amount"].sum()),
            high_severity_count=int((anom_subset["anomaly_severity"] == "High").sum()),
            medium_severity_count=int((anom_subset["anomaly_severity"] == "Medium").sum()),
            low_severity_count=int((anom_subset["anomaly_severity"] == "Low").sum()),
        )

        return df_out, summary
